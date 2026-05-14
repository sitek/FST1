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
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python modeling_firstlevel_GLMsingle.py --sub=FLT02 '
                        '--task=tonecat --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound '
                        '--t_acq=2 --t_r=3 '
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
                    help="what to model (options: `trial`, `sound`, or `stimulus`)", 
                    type=str)
parser.add_argument("--t_acq", 
                    help=("BOLD acquisition time (if different from "
                          "repetition time [TR], as in sparse designs)"), 
                    type=float)
parser.add_argument("--t_r", 
                    help="BOLD repetition time", 
                    type=float)
parser.add_argument("--bidsroot", 
                    help="top-level directory of the BIDS dataset", 
                    type=str)
parser.add_argument("--fmriprep_dir", 
                    help="directory of the fMRIprep preprocessed dataset", 
                    type=str)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    print(' ')
    sys.exit(1)
    
subject_id = args.sub
task_label = args.task
space_label=args.space
fwhm = args.fwhm
event_type=args.event_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# correct the fmriprep-given slice reference (middle slice, or 0.5)
# to account for sparse acquisition (silent gap during auditory presentation paradigm)
# fmriprep is explicitly based on slice timings, while nilearn is based on t_r
# and since images are only collected during a portion of the overall t_r 
# (which includes the silent gap),
# we need to account for this
slice_time_ref = 0.5 * t_acq / t_r

''' Pre-modeling functions '''
def update_events(models_events, event_type='sound'):
    ''' create events '''
    # stimulus events
    if event_type == 'stimulus':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')

                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])
    
    # trial-specific events
    if event_type == 'trial':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):

                name_groups = run_events.groupby('trial_type')['trial_type']
                suffix = name_groups.cumcount() + 1
                repeats = name_groups.transform('size')

                run_events['trial_type'] = run_events['trial_type'] + \
                                                    '_trial' + suffix.map(str)
                run_events['trial_type'] = run_events['trial_type'].str.replace('-','_')
                
                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([s for s in run_events['trial_type'].unique() if str(s) != 'nan'])

    # all sound events
    if event_type == 'sound':
        for sx, sub_events in enumerate(models_events):
            for mx, run_events in enumerate(sub_events):
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
                print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str.split('_', 
                                                                           expand=True)[0]

                # remove NaNs
                run_events.dropna(subset=['onset'], inplace=True)

        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
        print('stim list: ', stim_list)
        
    return stim_list, models_events

def convert_events_by_condition(models_events, tr, n_trs_per_run):
    """
    Convert models_events into GLMsingle-compatible design matrices
    grouped by 'sound' conditions.

    Parameters:
    - models_events: list of list of pd.DataFrame
    - tr: float, repetition time
    - n_trs_per_run: int, number of TRs per run

    Returns:
    - List of np.ndarray, each shaped [n_TRs × n_conditions]
    - List of condition names per run
    """
    glmsingle_matrices = []
    condition_names_per_run = []

    for subject_runs in models_events:
        for run_events in subject_runs:
            # Filter to only 'sound' trials
            sound_mask = run_events['trial_type'].str.contains('sound')
            sound_events = run_events[sound_mask]
            condition_names = sorted(sound_events['trial_type'].unique())

            # Initialize design matrix
            design_matrix = np.zeros((n_trs_per_run, len(condition_names)))

            # Fill matrix: 1 at onset TR for each condition
            for _, row in sound_events.iterrows():
                onset_tr = int(np.round(row['onset'] / tr))
                if onset_tr < n_trs_per_run:
                    cond_idx = condition_names.index(row['trial_type'])
                    design_matrix[onset_tr, cond_idx] = 1

            glmsingle_matrices.append(design_matrix)
            condition_names_per_run.append(condition_names)

    return glmsingle_matrices, condition_names_per_run

''' Start the modeling pipeline '''
# define bids and fmriprep directories
print('bidsroot: ', bidsroot)
print('fmriprep dir:', fmriprep_dir)

# create output directory
bidsderiv_dir = os.path.join(bidsroot, 
                             'derivatives', 
                             'glmsingle', 
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


# create updated events dataframes
stim_list, models_events = update_events(raw_models_events, 
                                         event_type=event_type)


# create a directory for saving GLMsingle outputs
outputdir_glmsingle = os.path.join(bidsderiv_dir, f'sub-{subject_id}')

opt = dict()

# set important fields for completeness (but these would be enabled by default)
opt['wantlibrary'] = 1
opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1

# for the purpose of this example we will keep the relevant outputs in memory
# and also save them to the disk
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [1,1,1,1]

# running python GLMsingle involves creating a GLM_single object
# and then running the procedure using the .fit() routine
glmsingle_obj = GLM_single(opt)

# visualize all the hyperparameters
print(glmsingle_obj.params)

data = [nib.load(x).get_fdata() for x in models_run_imgs[0]]
design = models_events[0]
stimdur = 0.3

tr = t_r
n_trs_per_run = data[0].shape[3]

xyz = data[0].shape[:3]

print('TR (in s):', t_r)
print('# TRs:', n_trs_per_run)

# convert the event dataframes into GLMsingle-compatible event matrices
glmsingle_matrices, condition_names = convert_events_by_condition(models_events, t_r, n_trs_per_run)

start_time = time.time()

if not os.path.exists(outputdir_glmsingle):

    print(f'running GLMsingle...')
    
    # run GLMsingle
    results_glmsingle = glmsingle_obj.fit(
       glmsingle_matrices,
       data,
       stimdur,
       tr,
       outputdir=outputdir_glmsingle)
    
    # we assign outputs of GLMsingle to the "results_glmsingle" variable.
    # note that results_glmsingle['typea'] contains GLM estimates from an ONOFF model,
    # where all images are treated as the same condition. these estimates
    # could be potentially used to find cortical areas that respond to
    # visual stimuli. we want to compare beta weights between conditions
    # therefore we are not going to include the ONOFF betas in any analyses of 
    # voxel reliability
    
else:
    print('GLMsingle outputs already exist:\n', outputdir_glmsingle)
    '''
    print(f'loading existing GLMsingle outputs from directory:\n\t{outputdir_glmsingle}')
    
    # load existing file outputs if they exist
    results_glmsingle = dict()
    results_glmsingle['typea'] = np.load(join(outputdir_glmsingle,
                                              'TYPEA_ONOFF.npy'),allow_pickle=True).item()
    results_glmsingle['typeb'] = np.load(join(outputdir_glmsingle,
                                              'TYPEB_FITHRF.npy'),allow_pickle=True).item()
    results_glmsingle['typec'] = np.load(join(outputdir_glmsingle,
                                              'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
    results_glmsingle['typed'] = np.load(join(outputdir_glmsingle,
                                              'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()
    '''
elapsed_time = time.time() - start_time

print(
    '\telapsed time: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)

