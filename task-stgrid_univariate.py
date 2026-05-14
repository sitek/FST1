import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from glob import glob
from nilearn import plotting

''' Set up and interpret command line arguments '''
parser = argparse.ArgumentParser(
                description='Subject-level modeling of fmriprep-preprocessed data',
                epilog=('Example: python univariate_analysis.py --sub=FLT02 '
                        '--task=stgrid --space=MNI152NLin2009cAsym '
                        '--fwhm=3 --event_type=sound --t_acq=2 --t_r=4 '
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
parser.add_argument("--model_type", 
                    help="trial model scheme (options: `LSA` or `LSS`)", 
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
model_type=args.model_type
t_acq = args.t_acq
t_r = args.t_r
bidsroot = args.bidsroot
fmriprep_dir = args.fmriprep_dir

# ## nilearn modeling: first level
# based on: https://nilearn.github.io/auto_examples/04_glm_first_level/plot_bids_features.html#sphx-glr-auto-examples-04-glm-first-level-plot-bids-features-py

def prep_models_and_args(subject_id=None, task_id=None, fwhm=None, bidsroot=None, 
                         fmriprep_dir=None, event_type=None, t_r=None, t_acq=None, space_label='T1w'):
    from nilearn.glm.first_level import first_level_from_bids
    from nilearn.interfaces.fmriprep import load_confounds

    data_dir = bidsroot

    task_label = task_id
    fwhm_sub = fwhm

    # correct the fmriprep-given slice reference (middle slice, or 0.5)
    # to account for sparse acquisition (silent gap during auditory presentation paradigm)
    # fmriprep is explicitly based on slice timings, while nilearn is based on t_r
    # and since images are only collected during a portion of the overall t_r (which includes the silent gap),
    # we need to account for this
    slice_time_ref = 0.5 * t_acq / t_r

    print(data_dir, task_label, space_label)

    models, models_run_imgs, \
            models_events, \
            models_confounds = first_level_from_bids(data_dir, task_label, space_label,
                                                     [subject_id],
                                                     smoothing_fwhm=fwhm,
                                                     derivatives_folder=fmriprep_dir,
                                                     slice_time_ref=slice_time_ref,
                                                     minimize_memory=False)

    # fill n/a with 0
    [[mc.fillna(0, inplace=True) for mc in sublist] for sublist in models_confounds]

    # define which confounds to keep as nuisance regressors
    conf_keep_list = ['framewise_displacement',
                      'trans_x', 'trans_y', 'trans_z', 
                      'rot_x','rot_y', 'rot_z',
                     ]
    '''
    # create stimulus list from events.tsv file
    if event_type=='block_stim':
        stim_list = sorted([str(s) for s in models_events[0][0]['trial_type'].unique() if str(s) not in ['nan', 'None']])
    elif event_type=='sound':
        stim_list = sorted([str(s)[:4] for s in models_events[0][0]['trial_type'].unique() if str(s) not in ['nan', 'None']])
    '''
    
    ''' create events '''
    for sx, sub_events in enumerate(models_events):        
        for mx, run_events in enumerate(sub_events):
            # stimulus events
            if event_type == 'block_stim':
                run_events['trial_type'] = run_events['trial_type']

            # combine all sound events
            elif event_type == 'sound':
                orig_stim_list = sorted([str(s) for s in run_events['trial_type'].unique() 
                                         if str(s) not in ['nan', 'None', 'null']])
                #print('original stim list: ', orig_stim_list)

                run_events['trial_type'] = run_events.trial_type.str[:4]

            # re-assign to models_events
            models_events[sx][mx] = run_events
            
        # create stimulus list from updated events.tsv file
        stim_list = sorted([str(s) for s in run_events['trial_type'].unique() if str(s) not in ['nan', 'None']])
    
    print('stim list: ', stim_list)
    return stim_list, models, models_run_imgs, models_events, models_confounds, conf_keep_list

# transform full event design matrix (LSA) into single-event only (LSS)
def lss_transformer(event_df, event_name):
    other_idx = np.array(event_df.loc[:,'trial_type'] != event_name)
    lss_event_df = event_df.copy()
    lss_event_df.loc[other_idx, 'trial_type'] = 'other_events' 
    return lss_event_df

# ### Across-runs GLM
def nilearn_glm_across_runs(stim_list, task_label, model_type, \
                            models, models_run_imgs, \
                            models_events, models_confounds, \
                            conf_keep_list, space_label):
    from nilearn.reporting import make_glm_report
    for midx in range(len(models)):
        for sx, stim in enumerate(stim_list):
            contrast_label = stim
            contrast_desc  = stim


            midx = 0
            model = models[midx]
            imgs = models_run_imgs[midx]
            #events = models_events[midx]
            confounds = models_confounds[midx]
            if model_type == 'LSA':
                events = models_events[midx]
            elif model_type == 'LSS':
                events = [lss_transformer(models_events[midx][rx], stim) for rx in range(len(imgs))]
            
            print(model.subject_label)

            # set limited confounds
            print('selecting confounds')
            confounds_ltd = [models_confounds[midx][cx][conf_keep_list] for cx in range(len(models_confounds[midx]))]
            
            #try:
            # fit the GLM
            print('fitting GLM')
            model.fit(imgs, events, confounds_ltd);

            # compute the contrast of interest
            print('computing contrast of interest')
            summary_statistics = model.compute_contrast(contrast_label, output_type='all')

            # prepare to save stat maps
            print('saving stat maps')

            from nilearn.interfaces.bids import save_glm_to_bids
            bidsderiv_sub_dir = os.path.join(bidsroot, 'derivatives', 'nilearn', 
                                             'bids-deriv_level-1_fwhm-%.02f'%model.smoothing_fwhm, 
                                             f'sub-{model.subject_label}_space-{space_label}',
                                             f'run-all_event-{event_type}')
            if not os.path.exists(bidsderiv_sub_dir):
                os.makedirs(bidsderiv_sub_dir)

            out_prefix = f"sub-{model.subject_label}_task-{task_label}_fwhm-{model.smoothing_fwhm}"
            save_glm_to_bids(model, 
                             contrast_label,
                             out_dir=bidsderiv_sub_dir,
                             prefix=out_prefix,
                            )
            print(f'Saved model outputs to {bidsderiv_sub_dir}')

            #except:
            #    print('could not run for ', contrast_label)
    return bidsderiv_sub_dir

nilearn_dir = os.path.join(bidsroot, 'derivatives', 'nilearn')
if not os.path.exists(nilearn_dir):
        os.makedirs(nilearn_dir)
        
stim_list, models, models_run_imgs, \
    models_events, models_confounds, \
    conf_keep_list = prep_models_and_args(subject_id, task_label, 
                                          fwhm, bidsroot, 
                                          fmriprep_dir, event_type,
                                          t_r, t_acq, 
                                          space_label)
# Across-run GLM
out_dir = nilearn_glm_across_runs(stim_list, task_label, 
                                  model_type, models, 
                                  models_run_imgs, 
                                  models_events, models_confounds, 
                                  conf_keep_list, space_label)
