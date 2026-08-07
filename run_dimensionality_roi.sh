#!/bin/bash

#SBATCH --time=0:30:00
#SBATCH -c 1

bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/
python dimensionality_roi.py --sub=$1 --bidsroot=${bidsroot}
