import os
import argparse
import sys

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from glob import glob
from nilearn.maskers import NiftiMasker
from nilearn import plotting, image


''' Command line arguments '''
parser = argparse.ArgumentParser(
    description='Voxelwise spectrotemporal modulation tuning analysis for stgrid task',
    epilog=('Example: python analyze_stgrid_spectrotemporal.py '
            '--bidsroot=/PATH/TO/BIDS/ '
            '--mask_dir=/PATH/TO/MASKS/ '
            '--grid_txt=2022-2-8_Grid.txt '
            '--out_dir=/PATH/TO/OUTPUT/')
)
parser.add_argument('--bidsroot', help='top-level BIDS directory', type=str)
parser.add_argument('--mask_dir',
                    help='directory containing per-subject MNI auditory cortex masks',
                    type=str)
parser.add_argument('--grid_txt', help='path to 2022-2-8_Grid.txt', type=str,
                    default='2022-2-8_Grid.txt')
parser.add_argument('--out_dir', help='output directory for tuning maps and figures', type=str)
parser.add_argument('--space', help='MNI space label', type=str,
                    default='MNI152NLin2009cAsym')
parser.add_argument('--sub', help='single subject to process (omit for all)', type=str,
                    default=None)

args = parser.parse_args()

if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

bidsroot  = args.bidsroot
mask_dir  = args.mask_dir
grid_txt  = args.grid_txt
out_dir   = args.out_dir
space     = args.space
sub_filter = args.sub

os.makedirs(out_dir, exist_ok=True)

glmsingle_dir = os.path.join(bidsroot, 'derivatives', 'glmsingle_stgrid',
                              'subject_level')

# auditory cortex ROI labels from MNI dseg (bilateral HG, PT, PP, STGa, STGp)
AUD_ROIS = ['L-HG', 'L-PT', 'L-PP', 'L-STGa', 'L-STGp',
            'R-HG', 'R-PT', 'R-PP', 'R-STGa', 'R-STGp']


''' Step 1 — Load stimulus grid '''
grid = pd.read_csv(grid_txt, sep='\t', header=None,
                   names=['idx', 'temporal_hz', 'spectral_coct'])
temporal_rates = sorted(grid['temporal_hz'].unique())    # [1.6, 6.07, 10.53, 15]
spectral_rates = sorted(grid['spectral_coct'].unique())  # [0.16, 0.94, 1.72, 2.5]
n_t = len(temporal_rates)
n_s = len(spectral_rates)

# map condition name (stim01 etc.) to (temporal_idx, spectral_idx) in the 4x4 grid
stim_to_grid = {}
for _, row in grid.iterrows():
    cond = f'stim{int(row.idx):02d}'
    t_idx = temporal_rates.index(row.temporal_hz)
    s_idx = spectral_rates.index(row.spectral_coct)
    stim_to_grid[cond] = (t_idx, s_idx)

print('Temporal modulation rates (Hz):', temporal_rates)
print('Spectral modulation rates (cyc/oct):', spectral_rates)
print('Stimulus → grid index mapping:', stim_to_grid)


''' Helper functions '''
def build_auditory_mask(subject_id, mask_dir, space, roi_labels):
    """Combine per-ROI binary masks into a single bilateral auditory cortex mask."""
    mask_imgs = []
    for roi in roi_labels:
        fpath = os.path.join(mask_dir,
                             f'sub-{subject_id}',
                             f'space-{space}',
                             'masks-dseg',
                             f'sub-{subject_id}_space-{space}_mask-{roi}.nii.gz')
        if os.path.exists(fpath):
            mask_imgs.append(nib.load(fpath))
        else:
            print(f'  WARNING: mask not found: {fpath}')
    if not mask_imgs:
        return None
    combined = image.math_img(
        'np.sum(imgs, axis=-1) > 0',
        imgs=image.concat_imgs(mask_imgs)
    )
    return combined


def load_beta_4d(subject_id, glmsingle_dir):
    """
    Load per-condition beta nii.gz files and stack into (X, Y, Z, 16) array.
    Returns (4D array, list of condition names, reference nibabel image).
    """
    beta_dir = os.path.join(glmsingle_dir, f'sub-{subject_id}', 'beta_images')
    beta_files = sorted(glob(os.path.join(beta_dir, 'di*_stim*.nii.gz')))
    if not beta_files:
        print(f'  No beta images found in {beta_dir}')
        return None, None, None
    ref_img = nib.load(beta_files[0])
    cond_names = [os.path.basename(f).split('_', 1)[1].replace('.nii.gz', '')
                  for f in beta_files]
    betas_4d = np.stack([nib.load(f).get_fdata() for f in beta_files], axis=-1)
    return betas_4d, cond_names, ref_img


def compute_tuning_maps(betas_4d, cond_names, stim_to_grid,
                        temporal_rates, spectral_rates):
    """
    For each voxel compute preferred temporal/spectral modulation rate
    and selectivity indices.

    betas_4d: (X, Y, Z, n_conditions)
    Returns dict of (X, Y, Z) arrays.
    """
    X, Y, Z, N = betas_4d.shape
    n_t = len(temporal_rates)
    n_s = len(spectral_rates)

    # reorder conditions to match grid (stim01–stim16 sorted = correct order)
    sorted_conds = sorted(cond_names, key=lambda c: stim_to_grid[c])
    cond_order = [cond_names.index(c) for c in sorted_conds]
    betas_ordered = betas_4d[..., cond_order]  # (X, Y, Z, 16) in grid order

    # reshape to (X, Y, Z, n_temporal, n_spectral)
    betas_grid = betas_ordered.reshape(X, Y, Z, n_t, n_s)

    # marginals
    temporal_marginal = betas_grid.mean(axis=4)  # (X, Y, Z, n_t)
    spectral_marginal = betas_grid.mean(axis=3)  # (X, Y, Z, n_s)

    t_arr = np.array(temporal_rates)
    s_arr = np.array(spectral_rates)

    # positive-rectify weights to avoid negative-weight artifacts
    def weighted_avg(marginal, values):
        weights = np.maximum(marginal, 0)
        w_sum = weights.sum(axis=-1, keepdims=True)
        w_sum = np.where(w_sum == 0, 1, w_sum)
        return (weights * values).sum(axis=-1) / w_sum.squeeze(-1)

    pref_temporal = weighted_avg(temporal_marginal, t_arr)   # (X, Y, Z)
    pref_spectral = weighted_avg(spectral_marginal, s_arr)   # (X, Y, Z)

    def selectivity(marginal):
        mx = marginal.max(axis=-1)
        mn = marginal.min(axis=-1)
        denom = np.abs(mx) + np.abs(mn)
        return np.where(denom == 0, 0, (mx - mn) / denom)

    sel_temporal = selectivity(temporal_marginal)  # (X, Y, Z)
    sel_spectral = selectivity(spectral_marginal)  # (X, Y, Z)

    return {
        'pref_temporal': pref_temporal,
        'pref_spectral': pref_spectral,
        'sel_temporal':  sel_temporal,
        'sel_spectral':  sel_spectral,
    }


def save_map(data, ref_img, fpath):
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(img, fpath)
    print(f'  saved {fpath}')


''' Step 2–4: Per-subject processing '''
sub_dirs = sorted(glob(os.path.join(glmsingle_dir, 'sub-*')))
if sub_filter:
    sub_dirs = [d for d in sub_dirs if os.path.basename(d) == f'sub-{sub_filter}']

subject_map_fpaths = {k: [] for k in ['pref_temporal', 'pref_spectral',
                                       'sel_temporal', 'sel_spectral']}

for sub_dir in sub_dirs:
    subject_id = os.path.basename(sub_dir).replace('sub-', '')
    print(f'\n--- Processing sub-{subject_id} ---')

    betas_4d, cond_names, ref_img = load_beta_4d(subject_id, glmsingle_dir)
    if betas_4d is None:
        print('  Skipping — no beta images')
        continue

    aud_mask = build_auditory_mask(subject_id, mask_dir, space, AUD_ROIS)
    if aud_mask is None:
        print('  Skipping — no auditory cortex mask')
        continue

    # apply mask (zero out non-auditory voxels)
    mask_data = image.resample_to_img(aud_mask, ref_img,
                                      interpolation='nearest').get_fdata().astype(bool)
    betas_masked = betas_4d.copy()
    betas_masked[~mask_data] = 0.0

    maps = compute_tuning_maps(betas_masked, cond_names, stim_to_grid,
                               temporal_rates, spectral_rates)

    # save per-subject maps
    sub_out = os.path.join(out_dir, f'sub-{subject_id}')
    os.makedirs(sub_out, exist_ok=True)
    for map_name, map_data in maps.items():
        fpath = os.path.join(sub_out,
                             f'sub-{subject_id}_task-stgrid_map-{map_name}.nii.gz')
        save_map(map_data, ref_img, fpath)
        subject_map_fpaths[map_name].append(fpath)


''' Step 5: Group average maps '''
print('\n--- Computing group average maps ---')
group_out = os.path.join(out_dir, 'group')
os.makedirs(group_out, exist_ok=True)

group_imgs = {}
for map_name, fpaths in subject_map_fpaths.items():
    if not fpaths:
        print(f'  No subject maps for {map_name}, skipping group average')
        continue
    imgs = [nib.load(f) for f in fpaths]
    group_mean = image.mean_img(imgs)
    group_fpath = os.path.join(group_out, f'group_task-stgrid_map-{map_name}.nii.gz')
    nib.save(group_mean, group_fpath)
    group_imgs[map_name] = group_mean
    print(f'  saved group map: {group_fpath}')


''' Step 6: Visualization '''
print('\n--- Generating figures ---')

fig_dir = os.path.join(out_dir, 'figures')
os.makedirs(fig_dir, exist_ok=True)

# preferred temporal modulation rate map
if 'pref_temporal' in group_imgs:
    display = plotting.plot_stat_map(
        group_imgs['pref_temporal'],
        title='Group preferred temporal modulation rate (Hz)',
        colorbar=True,
        cmap='RdYlBu_r',
        cut_coords=(-50, -20, 10),
        display_mode='ortho',
    )
    fig_fpath = os.path.join(fig_dir, 'group_task-stgrid_map-prefTemporal.png')
    display.savefig(fig_fpath)
    display.close()
    print(f'  saved {fig_fpath}')

# preferred spectral modulation rate map
if 'pref_spectral' in group_imgs:
    display = plotting.plot_stat_map(
        group_imgs['pref_spectral'],
        title='Group preferred spectral modulation rate (cyc/oct)',
        colorbar=True,
        cmap='RdYlGn',
        cut_coords=(-50, -20, 10),
        display_mode='ortho',
    )
    fig_fpath = os.path.join(fig_dir, 'group_task-stgrid_map-prefSpectral.png')
    display.savefig(fig_fpath)
    display.close()
    print(f'  saved {fig_fpath}')

# selectivity index maps
for map_name, title, cmap in [
    ('sel_temporal', 'Temporal modulation selectivity index', 'hot'),
    ('sel_spectral', 'Spectral modulation selectivity index', 'hot'),
]:
    if map_name in group_imgs:
        display = plotting.plot_stat_map(
            group_imgs[map_name],
            title=f'Group {title}',
            colorbar=True,
            cmap=cmap,
            vmin=0, vmax=1,
            cut_coords=(-50, -20, 10),
            display_mode='ortho',
        )
        fig_fpath = os.path.join(fig_dir, f'group_task-stgrid_map-{map_name}.png')
        display.savefig(fig_fpath)
        display.close()
        print(f'  saved {fig_fpath}')

print('\nDone.')
