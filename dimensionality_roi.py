#!/usr/bin/env python
# coding: utf-8

# Adapted from FLT2/dimensionality_roi.py for the STgrid task.
# STgrid is a single-session task (no learning-stage runs), so the stage-split
# logic from the FLT2 version (early/middle/final) is dropped here.

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd

from glob import glob
from collections import defaultdict

parser = argparse.ArgumentParser(
    description='Compute representational dimensionality (PR + eigenspectra) per ROI',
    epilog=('Example: python dimensionality_roi.py --sub=FLT02 '
            '--bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/')
)
parser.add_argument('--sub',      help='participant id', type=str, required=True)
parser.add_argument('--bidsroot', help='top-level BIDS directory', type=str, required=True)
args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

sub_id   = args.sub
bidsroot = args.bidsroot

glmsingle_dir = os.path.join(bidsroot, 'derivatives', 'glmsingle_stgrid')
out_dir       = os.path.join(glmsingle_dir, 'representational_dimensionality')
os.makedirs(out_dir, exist_ok=True)

# same ROI set/order as rsa_roi.py
roi_list = [
    'L-IC', 'L-MGN',
    'L-HG', 'L-PT', 'L-PP', 'L-STGp', 'L-STGa',
    'L-ParsOp', 'L-ParsTri',
    'R-IC', 'R-MGN',
    'R-HG', 'R-PT', 'R-PP', 'R-STGp', 'R-STGa',
    'R-ParsOp', 'R-ParsTri',
]

# STgrid's GLMsingle modeling produces one beta image per condition per run
# (stim01_run-01.nii.gz .. stim16_run-02.nii.gz for the 2 STgrid runs -- see
# save_betas_as_nifti in modeling_firstlevel_GLMsingle_stgrid.py). Repeated
# within-run presentations of a condition are already pooled into that one
# beta by the GLM design matrix, so there's no separate "rep" level -- just
# run and stimulus.
run_re  = re.compile(r'run-(\d+)')
stim_re = re.compile(r'(stim\d+)')


def load_roi_betas(sub_id, roi):
    """
    Load one trial beta vector per (run, stimulus) for one subject x ROI.

    Returns a nested dict: data[run_label][stim_label] = 1-D array
    """
    roi_folder = os.path.join(
        glmsingle_dir, 'masked_statmaps',
        f'sub-{sub_id}', 'statmaps_masked', f'mask-{roi}'
    )
    csv_files = sorted(glob(os.path.join(roi_folder, '*.csv')))
    if not csv_files:
        return None

    data = defaultdict(dict)
    for fpath in csv_files:
        fname  = os.path.basename(fpath)
        m_run  = run_re.search(fname)
        m_stim = stim_re.search(fname)
        if m_run is None or m_stim is None:
            continue
        try:
            vec = np.atleast_1d(np.genfromtxt(fpath))
        except Exception:
            continue
        if vec.ndim == 0 or np.all(np.isnan(vec)):
            continue
        data[f'run-{m_run.group(1)}'][m_stim.group(1)] = vec

    return data if data else None


def build_stimulus_matrix(data):
    """
    Given nested dict from load_roi_betas():
      1. Run de-mean: subtract each run's across-stimulus mean pattern
      2. Average the (now run-demeaned) pattern for each stimulus across runs
      3. Return (stimulus_labels, X) where X is (n_stimuli x n_voxels)
    """
    demeaned = defaultdict(list)
    for run_label, stim_dict in data.items():
        all_vecs = list(stim_dict.values())
        if not all_vecs:
            continue
        run_mean = np.mean(np.vstack(all_vecs), axis=0)
        for stim_label, vec in stim_dict.items():
            demeaned[stim_label].append(vec - run_mean)

    if not demeaned:
        return None, None

    rows, valid_labels = [], []
    for stim in sorted(demeaned.keys()):
        vecs = demeaned[stim]
        if not vecs:
            continue
        min_len = min(len(v) for v in vecs)
        rows.append(np.vstack([v[:min_len] for v in vecs]).mean(axis=0))
        valid_labels.append(stim)

    if len(rows) < 2:
        return None, None

    return valid_labels, np.vstack(rows)


def compute_dimensionality(X, standardize_voxels=True):
    n_stimuli, n_voxels = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    if standardize_voxels:
        col_std = X.std(axis=0, ddof=1)
        col_std[col_std < 1e-10] = 1.0
        X = X / col_std

    if n_stimuli <= n_voxels:
        G = X @ X.T / (n_stimuli - 1)
    else:
        G = X.T @ X / (n_stimuli - 1)

    eigenvalues = np.linalg.eigvalsh(G)
    eigenvalues = np.sort(eigenvalues[eigenvalues > 1e-10])[::-1]

    PR     = eigenvalues.sum()**2 / (eigenvalues**2).sum()
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()

    assert 1.0 <= PR <= min(n_stimuli, n_voxels) + 1e-6
    assert abs(cumvar[-1] - 1.0) < 1e-6

    # Sheng et al. (2022): N_k / prop_var_k  where k = eigenvalues >= 1
    kaiser_mask = eigenvalues >= 1
    N_k = int(kaiser_mask.sum())
    if N_k > 0:
        prop_var_k = eigenvalues[kaiser_mask].sum() / eigenvalues.sum()
        D_sheng = N_k / prop_var_k
    else:
        D_sheng = np.nan

    return {'PR': PR, 'D_sheng': D_sheng, 'eigenvalues': eigenvalues,
            'cumvar': cumvar, 'n_stimuli': n_stimuli, 'n_voxels': n_voxels}


# ---- Main loop ----
print(f'sub-{sub_id}')
sub_results = {}

for roi in roi_list:
    data = load_roi_betas(sub_id, roi)
    if data is None:
        print(f'  {roi}: no data')
        continue

    stim_labels, X = build_stimulus_matrix(data)
    if X is None:
        print(f'  {roi}: insufficient trials')
        continue

    try:
        result = compute_dimensionality(X)
    except AssertionError as e:
        print(f'  {roi}: sanity check failed — {e}')
        continue

    sub_results[roi] = result
    print(f'  {roi}: PR={result["PR"]:.2f}  D_sheng={result["D_sheng"]:.2f}  '
          f'({result["n_stimuli"]} stimuli, {result["n_voxels"]} voxels)')

if not sub_results:
    print('No results to save — exiting.')
    sys.exit(0)

# ---- Save ----
pr_row = {roi: res['PR'] for roi, res in sub_results.items()}
pd.DataFrame([pr_row]).to_csv(
    os.path.join(out_dir, f'sub-{sub_id}_dimensionality_PR.csv'),
    index=False
)

dsheng_row = {roi: res['D_sheng'] for roi, res in sub_results.items()}
pd.DataFrame([dsheng_row]).to_csv(
    os.path.join(out_dir, f'sub-{sub_id}_dimensionality_Dsheng.csv'),
    index=False
)

nvoxels_row = {roi: res['n_voxels'] for roi, res in sub_results.items()}
pd.DataFrame([nvoxels_row]).to_csv(
    os.path.join(out_dir, f'sub-{sub_id}_dimensionality_nvoxels.csv'),
    index=False
)

npz_data = {}
for roi, res in sub_results.items():
    npz_data[f'{roi}_eigenvalues'] = res['eigenvalues']
    npz_data[f'{roi}_cumvar']      = res['cumvar']
np.savez(
    os.path.join(out_dir, f'sub-{sub_id}_dimensionality_eigenspectra.npz'),
    **npz_data
)

print(f'Saved to {out_dir}')
