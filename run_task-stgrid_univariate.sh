#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32G

bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/

python task-stgrid_univariate.py --sub=$1 --task=stgrid \
  --space=MNI152NLin2009cAsym --fwhm=3 \
  --event_type=sound --model_type=LSA \
  --t_acq=2 --t_r=4 \
  --bidsroot=$bidsroot \
  --fmriprep_dir=$bidsroot/derivatives/denoised_fmriprep-22.1.1/