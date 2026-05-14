#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH -c 2

bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/
python rsa_roi.py --sub=$1 \
    --analysis_window=trial \
    --method=euclidean \
    --model=glmsingle \
    --mask_dir=${bidsroot}/derivatives/nilearn/masks/ \
    --bidsroot=${bidsroot} \
    --fmriprep_dir=${bidsroot}/derivatives/denoised_fmriprep-22.1.1/


