#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=48G

#conda activate py3
# fMRIPrep 25.2.5 / native T1w recompute (09_integration/INTEGRATION_PLAN.md, Workstream I,
# in fMRI_auditory-category-learning) -- writes to a version+space-tagged GLMsingle output
# dir so the original 22.1.1/MNI outputs under derivatives/glmsingle_stgrid/ are left untouched.
bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/

python modeling_firstlevel_GLMsingle_stgrid.py --sub=$1 --task=stgrid \
    --space=T1w --fwhm=0 \
    --event_type=block_stim --t_acq=2 --t_r=4 \
    --bidsroot=$bidsroot \
    --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-25.2.5/ \
    --glmsingle_dirname=glmsingle_stgrid_fmriprep-25.2.5_space-T1w
