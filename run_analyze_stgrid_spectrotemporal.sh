#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem=32G

bidsroot=/bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/
mask_dir=$bidsroot/derivatives/nilearn/masks/
out_dir=$bidsroot/derivatives/glmsingle_stgrid/spectrotemporal/
grid_txt=/bgfs/bchandrasekaran/krs228/data/FLT/code/FST1/2022-2-8_Grid.txt

# optional: pass subject id as first argument to run a single subject
sub_arg=""
if [ -n "$1" ]; then
    sub_arg="--sub=$1"
fi

python analyze_stgrid_spectrotemporal.py \
    --bidsroot=$bidsroot \
    --mask_dir=$mask_dir \
    --grid_txt=$grid_txt \
    --out_dir=$out_dir \
    --space=MNI152NLin2009cAsym \
    $sub_arg
