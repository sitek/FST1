#!/bin/bash
#SBATCH --time=16:00:00

# atlas options: 'tian-S2', 'tian-S3', 'dseg', 'subcort-aud'
# model options: 'stimulus_per_run_LSS', 'run-all_LSS', 'per_run_LSA_confound-compcor_event-stimulus'
#
# fMRIPrep 25.2.5 / native T1w recompute (09_integration/INTEGRATION_PLAN.md, Workstream I).
# --space=T1w reuses the ROI masks fMRI_auditory-category-learning/05_masking already warped
# into each subject's native T1w space: warp_atlases_to_T1w.sh + make_atlas_region_masks.py
# --space=T1w --atlas=carpet_dseg write exactly
#   $mask_dir/sub-<sub>/space-T1w/masks-dseg/sub-<sub>_space-T1w_mask-<roi>.nii.gz
# which is the same path/filename pattern this script already looks for -- no mask-path
# changes needed here, only --space and --glmsingle_dirname. NOTE: this assumes the tonecat
# and stgrid BOLD runs share the same acquisition matrix per subject -- these masks were
# resampled onto a tonecat functional reference image in 05_masking; if stgrid used a
# different matrix size, confirm the resample still lines up before trusting stgrid ROI betas.
bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/

model=glmsingle

# tian-S3 excluded: never warped to T1w (fMRI_auditory-category-learning/REVISION_PLAN.md,
# Workstream 8 -- "T1w+tian_S3 intentionally not built, out of scope"), so --space=T1w
# --atlas=tian-S3 has no mask directory to read from.
for atlas in dseg subcort-aud; do
python mask_glmsingle_betas.py --sub=$1 \
                           --fwhm=0.00 \
                           --atlas=$atlas \
                           --space=T1w \
                           --stat=t \
                           --model=$model \
                           --window=trial \
                            --mask_dir=$bidsroot/derivatives/nilearn/masks/ \
                            --bidsroot=$bidsroot \
                            --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-25.2.5/ \
                            --glmsingle_dirname=glmsingle_stgrid_fmriprep-25.2.5_space-T1w
done
