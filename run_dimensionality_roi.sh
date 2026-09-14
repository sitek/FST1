#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH -c 1

# fMRIPrep 25.2.5 / native T1w recompute (09_integration/INTEGRATION_PLAN.md, Workstream I)
bidsroot=/ix1/bchandrasekaran/krs228/data/FLT/data_denoised/
python dimensionality_roi.py --sub=$1 --bidsroot=${bidsroot} \
    --glmsingle_dirname=glmsingle_stgrid_fmriprep-25.2.5_space-T1w
