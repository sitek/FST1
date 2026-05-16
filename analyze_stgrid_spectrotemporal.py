import os
import argparse
import sys

import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from glob import glob
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

# subcortical ROIs: (label, atlas_dir_suffix)
SUBCORT_ROIS = [('L-IC',  'subcort-aud'),
                ('R-IC',  'subcort-aud'),
                ('L-MGN', 'subcort-aud'),
                ('R-MGN', 'subcort-aud')]

# cortical ROIs for per-ROI response surfaces
CORTEX_ROIS = [('L-HG',   'dseg'), ('R-HG',   'dseg'),
               ('L-PT',   'dseg'), ('R-PT',   'dseg'),
               ('L-STGp', 'dseg'), ('R-STGp', 'dseg'),
               ('L-STGa', 'dseg'), ('R-STGa', 'dseg')]


''' Step 1 — Load stimulus grid '''
grid = pd.read_csv(grid_txt, sep=r'\s+', header=None,
                   names=['temporal_hz', 'spectral_coct'])

print(grid)
print(grid.dtypes)

grid.dropna(inplace=True)

temporal_rates = sorted(grid['temporal_hz'].unique())    # [1.6, 6.07, 10.53, 15]
spectral_rates = sorted(grid['spectral_coct'].unique())  # [0.16, 0.94, 1.72, 2.5]
n_t = len(temporal_rates)
n_s = len(spectral_rates)

# map condition name (stim01 etc.) to (temporal_idx, spectral_idx) in the 4x4 grid
stim_to_grid = {}
for i, (_, row) in enumerate(grid.iterrows()):
    cond = f'stim{i+1:02d}'
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
    beta_files = sorted(glob(os.path.join(beta_dir, 'stim*.nii.gz')))
    if not beta_files:
        print(f'  No beta images found in {beta_dir}')
        return None, None, None
    ref_img = nib.load(beta_files[0])
    cond_names = [os.path.basename(f).replace('.nii.gz', '') for f in beta_files]
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

    # joint 2D center-of-mass: weights from full 4x4 surface simultaneously
    weights_2d = np.maximum(betas_grid, 0)  # (X, Y, Z, n_t, n_s)
    w2_sum = weights_2d.sum(axis=(3, 4), keepdims=True)
    w2_sum = np.where(w2_sum == 0, 1, w2_sum)
    joint_pref_temporal = (
        (weights_2d * t_arr[None, None, None, :, None]).sum(axis=(3, 4))
        / w2_sum.squeeze((3, 4))
    )
    joint_pref_spectral = (
        (weights_2d * s_arr[None, None, None, None, :]).sum(axis=(3, 4))
        / w2_sum.squeeze((3, 4))
    )

    return {
        'pref_temporal':       pref_temporal,
        'pref_spectral':       pref_spectral,
        'sel_temporal':        sel_temporal,
        'sel_spectral':        sel_spectral,
        'joint_pref_temporal': joint_pref_temporal,
        'joint_pref_spectral': joint_pref_spectral,
    }


def build_roi_mask(subject_id, mask_dir, space, roi_label, atlas):
    """Load a single ROI mask nii.gz; returns None if file is missing."""
    fpath = os.path.join(mask_dir,
                         f'sub-{subject_id}',
                         f'space-{space}',
                         f'masks-{atlas}',
                         f'sub-{subject_id}_space-{space}_mask-{roi_label}.nii.gz')
    if not os.path.exists(fpath):
        print(f'  WARNING: mask not found: {fpath}')
        return None
    return nib.load(fpath)


def compute_roi_response_surface(betas_4d, mask_img, ref_img,
                                 stim_to_grid, temporal_rates, spectral_rates):
    """
    Average betas across voxels within a ROI mask → (n_t, n_s) response surface.
    Returns None if the mask contains no voxels.
    """
    mask_data = image.resample_to_img(
        mask_img, ref_img, interpolation='nearest'
    ).get_fdata().astype(bool)
    betas_roi = betas_4d[mask_data]  # (n_voxels, n_conditions)
    if betas_roi.shape[0] == 0:
        return None

    n_t, n_s = len(temporal_rates), len(spectral_rates)
    sorted_conds = sorted(stim_to_grid.keys(), key=lambda c: stim_to_grid[c])
    # betas_roi columns are already sorted by condition name (stim01..stim16)
    # reorder to match grid layout
    cond_names_sorted = sorted(stim_to_grid.keys())
    grid_order = [cond_names_sorted.index(c) for c in sorted_conds]
    betas_ordered = betas_roi[:, grid_order]  # (n_voxels, 16)
    surface = betas_ordered.reshape(betas_roi.shape[0], n_t, n_s).mean(axis=0)
    return surface  # (n_t, n_s)


def save_map(data, ref_img, fpath):
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    nib.save(img, fpath)
    print(f'  saved {fpath}')


def plot_bivariate_map(pref_t_img, pref_s_img,
                       temporal_rates, spectral_rates,
                       title, out_fpath,
                       z_slices_mni=None):
    """
    Brain map where hue = preferred temporal rate, saturation = preferred
    spectral rate. Voxels with pref_temporal == 0 are treated as out-of-mask.
    Includes a 2-D color legend subplot.
    """
    pref_t = pref_t_img.get_fdata()
    pref_s = pref_s_img.get_fdata()
    affine = pref_t_img.affine
    inv_aff = np.linalg.inv(affine)

    mask = pref_t > 0

    t_min, t_max = temporal_rates[0], temporal_rates[-1]
    s_min, s_max = spectral_rates[0], spectral_rates[-1]

    h = np.clip((pref_t - t_min) / (t_max - t_min), 0, 1)
    s = np.clip((pref_s - s_min) / (s_max - s_min), 0, 1)
    v = np.ones_like(h)

    shape = h.shape
    hsv = np.stack([h, s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv.reshape(-1, 3)).reshape(*shape, 3)

    if z_slices_mni is None:
        z_slices_mni = [-20, -10, 0, 10, 20, 30]

    def mni_to_vox_z(mni_z):
        c = inv_aff @ np.array([0, 0, mni_z, 1])
        return int(np.round(c[2]))

    vox_zs = [mni_to_vox_z(z) for z in z_slices_mni]
    n = len(z_slices_mni)

    fig, axes = plt.subplots(1, n + 1, figsize=(3 * n + 3, 3.5))

    for ax, (mni_z, vox_z) in zip(axes[:n], zip(z_slices_mni, vox_zs)):
        ax.set_facecolor('k')
        if 0 <= vox_z < shape[2]:
            sl_rgb = rgb[:, :, vox_z, :]
            sl_mask = mask[:, :, vox_z]
            sl_rgba = np.zeros((*sl_rgb.shape[:2], 4), dtype=np.float32)
            sl_rgba[sl_mask, :3] = sl_rgb[sl_mask]
            sl_rgba[sl_mask, 3] = 1.0
            ax.imshow(np.rot90(sl_rgba), aspect='equal', interpolation='nearest')
        ax.set_title(f'z={mni_z}', fontsize=8)
        ax.axis('off')

    # 2-D color legend: x = temporal (hue), y = spectral (saturation)
    ax_leg = axes[-1]
    res = 64
    T2, S2 = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
    legend_hsv = np.stack([T2, S2, np.ones_like(T2)], axis=-1)
    legend_rgb = mcolors.hsv_to_rgb(legend_hsv)
    ax_leg.imshow(legend_rgb, origin='lower', aspect='auto',
                  extent=[t_min, t_max, s_min, s_max])
    ax_leg.set_xlabel('Temporal pref (Hz)', fontsize=8)
    ax_leg.set_ylabel('Spectral pref (cyc/oct)', fontsize=8)
    ax_leg.set_title('Color key', fontsize=8)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out_fpath}')


MAP_NAMES = ['pref_temporal', 'pref_spectral', 'sel_temporal', 'sel_spectral',
             'joint_pref_temporal', 'joint_pref_spectral']

if sub_filter:
    ''' Steps 2–4: Per-subject processing (runs when --sub is provided) '''
    sub_dirs = [os.path.join(glmsingle_dir, f'sub-{sub_filter}')]

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

        mask_data = image.resample_to_img(aud_mask, ref_img,
                                          interpolation='nearest').get_fdata().astype(bool)
        betas_masked = betas_4d.copy()
        betas_masked[~mask_data] = 0.0

        maps = compute_tuning_maps(betas_masked, cond_names, stim_to_grid,
                                   temporal_rates, spectral_rates)

        sub_out = os.path.join(out_dir, f'sub-{subject_id}')
        os.makedirs(sub_out, exist_ok=True)
        for map_name, map_data in maps.items():
            fpath = os.path.join(
                sub_out,
                f'sub-{subject_id}_task-stgrid_map-{map_name}.nii.gz'
            )
            save_map(map_data, ref_img, fpath)

        # per-ROI analysis: voxelwise maps + mean response surface
        all_rois = SUBCORT_ROIS + CORTEX_ROIS
        n_found = 0
        for roi_label, atlas in all_rois:
            roi_mask = build_roi_mask(subject_id, mask_dir, space,
                                      roi_label, atlas)
            if roi_mask is None:
                continue
            n_found += 1

            # voxelwise tuning maps within this ROI
            mask_data = image.resample_to_img(
                roi_mask, ref_img, interpolation='nearest'
            ).get_fdata().astype(bool)
            betas_roi_4d = betas_4d.copy()
            betas_roi_4d[~mask_data] = 0.0
            roi_maps = compute_tuning_maps(betas_roi_4d, cond_names,
                                           stim_to_grid,
                                           temporal_rates, spectral_rates)
            for map_name, map_data in roi_maps.items():
                fpath = os.path.join(
                    sub_out,
                    f'sub-{subject_id}_task-stgrid'
                    f'_roi-{roi_label}_map-{map_name}.nii.gz'
                )
                save_map(map_data, ref_img, fpath)

            # mean response surface (n_t x n_s) saved as CSV
            surface = compute_roi_response_surface(
                betas_4d, roi_mask, ref_img,
                stim_to_grid, temporal_rates, spectral_rates
            )
            if surface is not None:
                surf_df = pd.DataFrame(
                    surface,
                    index=temporal_rates,
                    columns=spectral_rates
                )
                surf_df.index.name = 'temporal_hz'
                surf_df.columns.name = 'spectral_coct'
                csv_fpath = os.path.join(
                    sub_out,
                    f'sub-{subject_id}_task-stgrid_roi-{roi_label}_surface.csv'
                )
                surf_df.to_csv(csv_fpath)
                print(f'  saved {csv_fpath}')

        print(f'  ROI masks found: {n_found}/{len(all_rois)}')

    print('\nPer-subject processing complete.')

else:
    ''' Step 5: Group average maps (runs when --sub is not provided) '''
    print('\n--- Computing group average maps ---')
    group_out = os.path.join(out_dir, 'group')
    os.makedirs(group_out, exist_ok=True)

    group_imgs = {}
    for map_name in MAP_NAMES:
        # use specific pattern to exclude per-ROI maps (which contain '_roi-')
        fpaths = sorted(glob(os.path.join(out_dir, 'sub-*',
                                          f'*_task-stgrid_map-{map_name}.nii.gz')))
        if not fpaths:
            print(f'  No subject maps found for {map_name}, skipping')
            continue
        print(f'  Averaging {len(fpaths)} subjects for {map_name}')
        group_mean = image.mean_img([nib.load(f) for f in fpaths])
        group_fpath = os.path.join(group_out, f'group_task-stgrid_map-{map_name}.nii.gz')
        nib.save(group_mean, group_fpath)
        group_imgs[map_name] = group_mean
        print(f'  saved {group_fpath}')

    # group averaging of per-ROI voxelwise maps (for IC, MGN, and cortical ROIs)
    print('\n--- Computing group per-ROI voxelwise maps ---')
    all_rois = SUBCORT_ROIS + CORTEX_ROIS
    group_roi_imgs = {}
    for roi_label, _ in all_rois:
        group_roi_imgs[roi_label] = {}
        for map_name in MAP_NAMES:
            fpaths = sorted(glob(os.path.join(
                out_dir, 'sub-*',
                f'*_task-stgrid_roi-{roi_label}_map-{map_name}.nii.gz'
            )))
            if not fpaths:
                continue
            print(f'  {roi_label}/{map_name}: {len(fpaths)} subjects')
            group_roi_mean = image.mean_img([nib.load(f) for f in fpaths])
            group_fpath = os.path.join(
                group_out,
                f'group_task-stgrid_roi-{roi_label}_map-{map_name}.nii.gz'
            )
            nib.save(group_roi_mean, group_fpath)
            group_roi_imgs[roi_label][map_name] = group_roi_mean
            print(f'    saved {group_fpath}')

    # group mean response surfaces per ROI
    print('\n--- Computing group ROI response surfaces ---')
    all_rois = SUBCORT_ROIS + CORTEX_ROIS
    group_surfaces = {}
    for roi_label, _ in all_rois:
        csv_fpaths = sorted(glob(os.path.join(
            out_dir, 'sub-*',
            f'*_roi-{roi_label}_surface.csv'
        )))
        if not csv_fpaths:
            print(f'  No surface CSVs for {roi_label}, skipping')
            continue
        surfaces = [pd.read_csv(f, index_col=0) for f in csv_fpaths]
        group_surf = pd.concat(surfaces).groupby(level=0).mean()
        group_surf_fpath = os.path.join(
            group_out,
            f'group_task-stgrid_roi-{roi_label}_surface.csv'
        )
        group_surf.to_csv(group_surf_fpath)
        group_surfaces[roi_label] = group_surf
        print(f'  saved {group_surf_fpath} ({len(csv_fpaths)} subjects)')


''' Step 6: Visualization (group mode only) '''
if sub_filter:
    print('\nDone.')
    sys.exit(0)

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

# bivariate joint spectrotemporal map (auditory cortex)
if 'joint_pref_temporal' in group_imgs and 'joint_pref_spectral' in group_imgs:
    plot_bivariate_map(
        group_imgs['joint_pref_temporal'],
        group_imgs['joint_pref_spectral'],
        temporal_rates, spectral_rates,
        title='Group joint spectrotemporal preference (hue=temporal, sat=spectral)',
        out_fpath=os.path.join(fig_dir, 'group_task-stgrid_map-bivariate.png'),
        z_slices_mni=[-20, -10, 0, 10, 20, 30],
    )

# bivariate maps for each subcortical ROI (zoomed z-slices)
SUBCORT_Z = {'L-IC': [-20, -16, -12], 'R-IC': [-20, -16, -12],
             'L-MGN': [-8, -4, 0],    'R-MGN': [-8, -4, 0]}
for roi_label in ['L-IC', 'R-IC', 'L-MGN', 'R-MGN']:
    if roi_label not in group_roi_imgs:
        continue
    ri = group_roi_imgs[roi_label]
    if 'joint_pref_temporal' not in ri or 'joint_pref_spectral' not in ri:
        continue
    plot_bivariate_map(
        ri['joint_pref_temporal'],
        ri['joint_pref_spectral'],
        temporal_rates, spectral_rates,
        title=f'{roi_label} joint spectrotemporal preference',
        out_fpath=os.path.join(
            fig_dir, f'group_task-stgrid_roi-{roi_label}_map-bivariate.png'),
        z_slices_mni=SUBCORT_Z[roi_label],
    )

# zoomed voxelwise plots for subcortical ROIs (IC and MGN)
# cut_coords are MNI centers: IC ~(0, -36, -14), MGN ~(0, -26, -4)
SUBCORT_PLOT_CFG = {
    'L-IC':  ((-4,  -36, -14), 'Inf colliculus (L)'),
    'R-IC':  (( 4,  -36, -14), 'Inf colliculus (R)'),
    'L-MGN': ((-14, -26,  -4), 'Med geniculate (L)'),
    'R-MGN': (( 14, -26,  -4), 'Med geniculate (R)'),
}
for map_name, title_suffix, cmap in [
    ('pref_temporal',       'pref temporal (Hz)',       'RdYlBu_r'),
    ('pref_spectral',       'pref spectral (cyc/oct)',  'RdYlGn'),
    ('joint_pref_temporal', 'joint pref temporal (Hz)', 'RdYlBu_r'),
    ('joint_pref_spectral', 'joint pref spectral (c/o)','RdYlGn'),
]:
    for roi_label, (cut_coords, roi_title) in SUBCORT_PLOT_CFG.items():
        if roi_label not in group_roi_imgs:
            continue
        if map_name not in group_roi_imgs[roi_label]:
            continue
        roi_img = group_roi_imgs[roi_label][map_name]
        display = plotting.plot_stat_map(
            roi_img,
            title=f'{roi_title} — {title_suffix}',
            colorbar=True,
            cmap=cmap,
            cut_coords=cut_coords,
            display_mode='ortho',
            annotate=True,
        )
        fig_fpath = os.path.join(
            fig_dir,
            f'group_task-stgrid_roi-{roi_label}_map-{map_name}.png'
        )
        display.savefig(fig_fpath, dpi=200)
        display.close()
        print(f'  saved {fig_fpath}')

# ROI response surface heatmaps
if group_surfaces:
    roi_order = [r for r, _ in SUBCORT_ROIS + CORTEX_ROIS
                 if r in group_surfaces]
    n_rois = len(roi_order)
    fig, axes = plt.subplots(1, n_rois, figsize=(3 * n_rois, 3.5),
                             constrained_layout=True)
    if n_rois == 1:
        axes = [axes]
    for ax, roi_label in zip(axes, roi_order):
        surf = group_surfaces[roi_label].values
        vmax = np.abs(surf).max()
        im = ax.imshow(surf, aspect='auto', origin='lower',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(roi_label, fontsize=9)
        ax.set_xlabel('Spectral (cyc/oct)', fontsize=7)
        ax.set_ylabel('Temporal (Hz)', fontsize=7)
        ax.set_xticks(range(len(spectral_rates)))
        ax.set_yticks(range(len(temporal_rates)))
        ax.set_xticklabels([f'{r:.2f}' for r in spectral_rates],
                           fontsize=6)
        ax.set_yticklabels([f'{r:.1f}' for r in temporal_rates],
                           fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.7, label='β')
    heatmap_fpath = os.path.join(fig_dir,
                                 'group_task-stgrid_roi-surfaces.png')
    fig.savefig(heatmap_fpath, dpi=150)
    plt.close(fig)
    print(f'  saved {heatmap_fpath}')

print('\nDone.')
