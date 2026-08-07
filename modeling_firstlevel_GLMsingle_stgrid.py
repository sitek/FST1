import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import os
import time
import argparse
import sys

from glmsingle import GLM_single
from glob import glob
from os.path import join, exists, split
from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.fmriprep import load_confounds


''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level GLMsingle modeling of stgrid fmriprep-preprocessed data',
                epilog=('Example: python modeling_firstlevel_GLMsingle_stgrid.py --sub=FLT02 '
                        '--task=stgrid --space=MNI152NLin2009cAsym '
                        '--fwhm=0 --event_type=block_stim '
                        '--t_acq=2 --t_r=4 '
                        '--bidsroot=/PATH/TO/BIDS/DIR/ '
                        '--fmriprep_dir=/PATH/TO/FMRIPREP/DIR/')
                )

parser.add_argument("--sub",
                    help="participant id", type=str)
parser.add_argument("--task",
                    help="task id", type=str)
parser.add_argument("--space",
                    help="space label", type=str)
parser.add_argument("--fwhm",
                    help="spatial smoothing full-width half-max",
                    type=float)
parser.add_argument("--event_type",
                    help="what to model (options: `block_stim` or `sound`)",
                    type=str)
parser.add_argument("--t_acq",
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"),
                    type=float)
parser.add_argument("--t_r",
                    help="BOLD repetition time", type=float)
parser.add_argument("--bidsroot",
                    help="top-level directory of the BIDS dataset", type=str)
parser.add_argument("--fmriprep_dir",
                    help="directory of the fMRIprep preprocessed dataset", type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)

subject_id   = args.sub
task_label   = args.task
space_label  = args.space
fwhm         = args.fwhm
event_type   = args.event_type
t_acq        = args.t_acq
t_r          = args.t_r
bidsroot     = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# correct the fmriprep-given slice reference (middle slice, or 0.5)
# to account for sparse acquisition (silent gap during auditory presentation paradigm)
slice_time_ref = 0.5 * t_acq / t_r


''' Pre-modeling functions '''
def update_events(models_events, event_type='block_stim'):
    if event_type == 'block_stim':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events.dropna(subset=['onset'], inplace=True)
        stim_list = sorted([s for s in run_events['trial_type'].unique()
                            if str(s) not in ['nan', 'None']])

    elif event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique()
                                         if str(s) not in ['nan', 'None']])
                print('original stim list: ', orig_stim_list)
                run_events['trial_type'] = run_events.trial_type.str.split('_',
                                                                           expand=True)[0]
                run_events.dropna(subset=['onset'], inplace=True)
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique()
                            if str(s) not in ['nan', 'None']])
        print('stim list: ', stim_list)

    return stim_list, models_events


def convert_events_to_glmsingle(models_events, tr, n_trs_per_run):
    """
    Convert models_events into GLMsingle-compatible design matrices
    for stgrid block_stim conditions (stim01–stim16).

    Returns:
    - List of np.ndarray, each shaped [n_TRs x n_conditions]
    - List of condition names per run
    """
    glmsingle_matrices = []
    condition_names_per_run = []

    for subject_runs in models_events:
        for run_events in subject_runs:
            stim_mask = run_events['trial_type'].str.startswith('stim')
            stim_events = run_events[stim_mask]
            condition_names = sorted(stim_events['trial_type'].unique())

            design_matrix = np.zeros((n_trs_per_run, len(condition_names)))

            for _, row in stim_events.iterrows():
                onset_tr = int(np.round(row['onset'] / tr))
                if onset_tr < n_trs_per_run:
                    cond_idx = condition_names.index(row['trial_type'])
                    design_matrix[onset_tr, cond_idx] = 1

            glmsingle_matrices.append(design_matrix)
            condition_names_per_run.append(condition_names)

    return glmsingle_matrices, condition_names_per_run


def save_betas_as_nifti(results, condition_names, ref_nib_img, outputdir):
    """Save GLMsingle TYPED beta estimates as one nii.gz per condition per run.

    `condition_names` is a list of lists -- one list of condition names per
    run, in the same run order the design matrices were passed to
    GLMsingle.fit(). GLMsingle flattens every run's design-matrix columns
    into betasmd's last axis in that same run-then-column order (see
    GLMsingle.ipynb's indexing scheme), so this must iterate runs then
    conditions to recover the correct (run, condition) label for each slice.
    """
    betas = results['typed']['betasmd']  # shape: (X, Y, Z, sum of columns across runs)
    n_expected = sum(len(names) for names in condition_names)
    assert betas.shape[-1] == n_expected, (
        f'betasmd has {betas.shape[-1]} volumes but condition_names implies '
        f'{n_expected} (across {len(condition_names)} runs) -- indexing would be wrong'
    )

    beta_dir = os.path.join(outputdir, 'beta_images')
    os.makedirs(beta_dir, exist_ok=True)
    affine = ref_nib_img.affine
    header = ref_nib_img.header

    idx = 0
    n_saved = 0
    for r, run_condition_names in enumerate(condition_names):
        for cond_name in run_condition_names:
            img = nib.Nifti1Image(betas[:, :, :, idx].astype(np.float32), affine, header)
            out_fpath = os.path.join(beta_dir, f'{cond_name}_run-{r+1:02d}.nii.gz')
            nib.save(img, out_fpath)
            print(f'  saved {out_fpath}')
            idx += 1
            n_saved += 1
    print(f'Saved {n_saved} beta images to {beta_dir}')


def save_r2_as_nifti(results, ref_nib_img, outputdir):
    """Save GLMsingle R² map (converted from 0–100 to 0–1 proportion)."""
    r2_raw = results['typed']['R2']   # (X, Y, Z), values 0–100
    r2 = (r2_raw / 100.0).astype(np.float32)
    img = nib.Nifti1Image(r2, ref_nib_img.affine, ref_nib_img.header)
    out_fpath = os.path.join(outputdir, 'R2.nii.gz')
    nib.save(img, out_fpath)
    print(f'  saved {out_fpath}')


''' Start the modeling pipeline '''
print('bidsroot: ', bidsroot)
print('fmriprep dir:', fmriprep_dir)

# output directory — stgrid-specific to avoid confusion with tonecat outputs
bidsderiv_dir = os.path.join(bidsroot,
                             'derivatives',
                             'glmsingle_stgrid',
                             'subject_level')
if not os.path.exists(bidsderiv_dir):
    os.makedirs(bidsderiv_dir)
print('bidsderiv dir:', bidsderiv_dir)

models, models_run_imgs, \
        raw_models_events, \
        models_confounds = first_level_from_bids(bidsroot,
                                                 task_label,
                                                 space_label=space_label,
                                                 sub_labels=[subject_id],
                                                 smoothing_fwhm=fwhm,
                                                 derivatives_folder=fmriprep_dir,
                                                 slice_time_ref=slice_time_ref,
                                                 minimize_memory=False)
print('models_run_imgs:', models_run_imgs)

# read stimulus duration from events.tsv (all 16 stim conditions share the same duration)
try:
    stim_rows = raw_models_events[0][0][
        raw_models_events[0][0]['trial_type'].str.startswith('stim', na=False)
    ]
    stimdur = float(stim_rows['duration'].iloc[0])
    print(f'stimdur from events.tsv: {stimdur}s')
except Exception as e:
    stimdur = 1.0
    print(f'Could not read stimdur from events ({e}), defaulting to {stimdur}s')

stim_list, models_events = update_events(raw_models_events, event_type=event_type)
print('stim list:', stim_list)

outputdir_glmsingle = os.path.join(bidsderiv_dir, f'sub-{subject_id}')

opt = dict()
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1
opt['wantfileoutputs'] = [1, 1, 1, 1]
opt['wantmemoryoutputs'] = [1, 1, 1, 1]

glmsingle_obj = GLM_single(opt)
print(glmsingle_obj.params)

data = [nib.load(x).get_fdata() for x in models_run_imgs[0]]
n_trs_per_run = data[0].shape[3]
tr = t_r

print('TR (in s):', tr)
print('# TRs:', n_trs_per_run)
print('stimdur (in s):', stimdur)

glmsingle_matrices, condition_names = convert_events_to_glmsingle(models_events, tr, n_trs_per_run)

# condition_names is a list-of-lists, one list per run, in the same run order
# as glmsingle_matrices/data -- this full structure (not just run 1) is what
# save_betas_as_nifti needs to correctly label every (run, condition) volume.
n_runs = len(condition_names)
n_betas_expected = sum(len(names) for names in condition_names)
print(f'conditions per run: {condition_names}')
print(f'{n_runs} runs, {n_betas_expected} beta volumes expected')

start_time = time.time()

if not os.path.exists(outputdir_glmsingle):
    print(f'running GLMsingle...')

    results_glmsingle = glmsingle_obj.fit(
        glmsingle_matrices,
        data,
        stimdur,
        tr,
        outputdir=outputdir_glmsingle)

    # save TYPED beta estimates and R² as nii.gz files
    ref_img = nib.load(models_run_imgs[0][0])
    save_betas_as_nifti(results_glmsingle, condition_names,
                        ref_img, outputdir_glmsingle)
    save_r2_as_nifti(results_glmsingle, ref_img, outputdir_glmsingle)

else:
    print('GLMsingle outputs already exist, loading from:\n', outputdir_glmsingle)
    results_glmsingle = dict()
    results_glmsingle['typea'] = np.load(join(outputdir_glmsingle,
                                              'TYPEA_ONOFF.npy'),
                                         allow_pickle=True).item()
    results_glmsingle['typeb'] = np.load(join(outputdir_glmsingle,
                                              'TYPEB_FITHRF.npy'),
                                         allow_pickle=True).item()
    results_glmsingle['typec'] = np.load(join(outputdir_glmsingle,
                                              'TYPEC_FITHRF_GLMDENOISE.npy'),
                                         allow_pickle=True).item()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle,
                                              'TYPED_FITHRF_GLMDENOISE_RR.npy'),
                                         allow_pickle=True).item()

    # save nii.gz betas and R² if not already done. Check the *count* of
    # existing files, not just directory existence -- a previous buggy
    # version of save_betas_as_nifti only ever exported run 1's volumes,
    # so a stale/incomplete beta_images/ dir must be regenerated here.
    ref_img = nib.load(models_run_imgs[0][0])
    beta_dir = os.path.join(outputdir_glmsingle, 'beta_images')
    existing_betas = glob(os.path.join(beta_dir, '*.nii.gz'))
    if len(existing_betas) != n_betas_expected:
        print(f'beta_images/ has {len(existing_betas)} files, expected {n_betas_expected} '
              '-- (re)exporting')
        save_betas_as_nifti(results_glmsingle, condition_names,
                            ref_img, outputdir_glmsingle)
    else:
        print('beta_images directory already complete, skipping nii.gz export')
    r2_path = os.path.join(outputdir_glmsingle, 'R2.nii.gz')
    if not os.path.exists(r2_path):
        save_r2_as_nifti(results_glmsingle, ref_img, outputdir_glmsingle)
    else:
        print('R2.nii.gz already exists, skipping')

elapsed_time = time.time() - start_time
print('\telapsed time: ',
      f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
