#!/usr/bin/env python
# coding: utf-8

# Based on the rsatoolbox tutorial: https://rsatoolbox.readthedocs.io/en/stable/demo_searchlight.html
import os
import argparse
import sys
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import seaborn as sns

from nilearn import plotting
from nilearn.image import new_img_like

import rsatoolbox
from rsatoolbox.inference import eval_fixed
from rsatoolbox.model import ModelFixed, Model
from rsatoolbox.rdm import RDMs

from rsatoolbox.util.searchlight import get_volume_searchlight, get_searchlight_RDMs, evaluate_models_searchlight
from glob import glob

parser = argparse.ArgumentParser(
                description='Create subject-specific GLMsingle RSA',
                epilog=('Example: python rsa_roi.py --sub=FLT02 '
                        ' --analysis_window=trial '
                        ' --method=euclidean '
                        ' --model=glmsingle '
                        ' --mask_dir=/PATH/TO/MASK/DIR/ '                        
                        ' --bidsroot=/PATH/TO/BIDS/DIR/ ' 
                        ' --fmriprep_dir=/PATH/TO/FMRIPREP/DIR/'))

parser.add_argument("--sub", help="participant id", 
                    type=str)
parser.add_argument("--analysis_window", 
                    help="analysis window (options: session, run}", 
                    type=str)
parser.add_argument("--method", 
                    help="calculation method (options: crossnobis, euclidean, correlation}", 
                    type=str)
parser.add_argument("--model", 
                    help=("which model to operate on "
                          "(options: glmsingle)"), 
                    type=str)
parser.add_argument("--mask_dir", 
                    help="directory containing subdirectories with masks for each subject", 
                    type=str)
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
    
sub_id          = args.sub
analysis_window = args.analysis_window
method_label    = args.method
model_desc      = args.model
mask_dir        = args.mask_dir
bidsroot        = args.bidsroot
fmriprep_dir    = args.fmriprep_dir

# other directory definitions
deriv_dir = os.path.join(bidsroot, 'derivatives')
model_dir = os.path.join(deriv_dir, 'glmsingle')

# method
print('RDM calculation method: ', method_label)

# define ROIs
network_name = 'auditory'

roi_list = [
            'L-IC', 'L-MGN', 
            'L-HG', 'L-PT',  'L-PP', 'L-STGp', 'L-STGa', 
            'L-ParsOp', 'L-ParsTri',
            'R-IC', 'R-MGN', 
            'R-HG', 'R-PT',  'R-PP', 'R-STGp', 'R-STGa', 
            'R-ParsOp', 'R-ParsTri', 
           ]

''' Generate trial-specific RDMs '''
assert analysis_window == 'trial'

model_folder = os.path.join(model_dir,
                            'masked_statmaps',
                            f'sub-{sub_id}',
                            'statmaps_masked'
                        )
print('model_folder:', model_folder)

roi_rdm_list = []

# ---- regex patterns ----
run_re  = re.compile(r'run-(\d+)')
stim_re = re.compile(r'(di\d+_[A-Za-z]+)')
rep_re  = re.compile(r'rep-(\d+)')

'''
for roi in roi_list:
    roi_folder = os.path.join(model_folder, f'mask-{roi}')
    csv_files = sorted(glob(os.path.join(roi_folder, '*.csv')))

    if len(csv_files) == 0:
        print(f'No files found for ROI {roi}')
        continue

    data_list = []
    obs_desc = {
        'run': [],
        'stimulus': [],
        'rep': []
    }

    # ---- loop over ALL files (all runs) ----
    for f in csv_files:
        fname = os.path.basename(f)

        m_run  = run_re.search(fname)
        stim   = stim_re.search(fname)
        rep    = rep_re.search(fname)

        if m_run is None or stim is None or rep is None:
            continue

        run_label  = f'run-{m_run.group(1)}'
        stim_label = stim.group(1)
        rep_label  = f'rep-{rep.group(1)}'

        try:
            vec = np.genfromtxt(f)
        except Exception:
            continue

        vec = np.atleast_1d(vec)

        data_list.append(vec)
        obs_desc['run'].append(run_label)
        obs_desc['stimulus'].append(stim_label)
        obs_desc['rep'].append(rep_label)

    if len(data_list) < 2:
        print(f'Skipping ROI {roi} (not enough trials)')
        continue

    data = np.vstack(data_list)

    # ---- ONE dataset per ROI, across runs ----
    dataset = rsatoolbox.data.Dataset(
        data,
        descriptors={
            'participant': sub_id,
            'ROI': roi
        },
        obs_descriptors=obs_desc
    )

    # ---- crossvalidated RDM ----
    rdm = rsatoolbox.rdm.calc_rdm(
        dataset,
        method=method_label,
        descriptor='stimulus',
        cv_descriptor='run'
    )

    roi_rdm_list.append(rdm)
'''

for roi in roi_list:
    roi_folder = os.path.join(model_folder, f'mask-{roi}')
    csv_files = sorted(glob(os.path.join(roi_folder, '*.csv')))

    if len(csv_files) == 0:
        print(f'No files found for ROI {roi}')
        continue

    # group files by run
    run_files = {}
    for f in csv_files:
        fname  = os.path.basename(f)
        m_run  = run_re.search(fname)
        stim   = stim_re.search(fname)
        rep    = rep_re.search(fname)

        if m_run is None or stim is None or rep is None:
            continue

        run_label = f'run-{m_run.group(1)}'
        run_files.setdefault(run_label, []).append((f, stim.group(1), f'rep-{rep.group(1)}'))

    if method_label == 'crossnobis':
        data_list = []
        obs_desc  = {'run': [], 'stimulus': [], 'rep': []}

        for run_label, file_entries in sorted(run_files.items()):
            for f, stim_label, rep_label in file_entries:
                try:
                    vec = np.atleast_1d(np.genfromtxt(f))
                except Exception:
                    continue
                data_list.append(vec)
                obs_desc['run'].append(run_label)
                obs_desc['stimulus'].append(stim_label)
                obs_desc['rep'].append(rep_label)

        if len(data_list) < 2:
            print(f'Skipping ROI {roi} (not enough trials)')
            continue

        dataset = rsatoolbox.data.Dataset(
            np.vstack(data_list),
            descriptors={'participant': sub_id, 'ROI': roi},
            obs_descriptors=obs_desc
        )
        rdm = rsatoolbox.rdm.calc_rdm(
            dataset, method=method_label, descriptor='stimulus', cv_descriptor='run'
        )
        roi_rdm_list.append(rdm)

    else:
        for run_label, file_entries in sorted(run_files.items()):
            data_list = []
            obs_desc  = {'run': [], 'stimulus': [], 'rep': []}

            for f, stim_label, rep_label in file_entries:
                try:
                    vec = np.atleast_1d(np.genfromtxt(f))
                except Exception:
                    continue
                data_list.append(vec)
                obs_desc['run'].append(run_label)
                obs_desc['stimulus'].append(stim_label)
                obs_desc['rep'].append(rep_label)

            if len(data_list) < 2:
                print(f'Skipping ROI {roi}, {run_label} (not enough trials)')
                continue

            dataset = rsatoolbox.data.Dataset(
                np.vstack(data_list),
                descriptors={'participant': sub_id, 'ROI': roi, 'run': run_label},
                obs_descriptors=obs_desc
            )
            rdm = rsatoolbox.rdm.calc_rdm(
                dataset, method=method_label, descriptor='stimulus'
            )
            roi_rdm_list.append(rdm)

concat_rdms = rsatoolbox.rdm.rdms.concat(roi_rdm_list)
concat_rdms.descriptors['participant'] = sub_id

# save subject-level RDMs
out_dir = os.path.join(model_dir, f'rsa-roi_{model_desc}_rdmcalc-{method_label}')
os.makedirs(out_dir, exist_ok=True)
basename = f'sub-{sub_id}_{model_desc}_{analysis_window}_{network_name}_{method_label}_rdms'
out_fpath = os.path.join(out_dir,
                         f'{basename}.hdf5')
concat_rdms.save(out_fpath, 
                 file_type='hdf5', overwrite=True)
print('saved RDMs to', out_fpath)


