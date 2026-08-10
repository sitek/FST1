#!/bin/bash

for subpath in /ix1/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do
  fullsubid=$(basename $subpath)
  subid=${fullsubid#sub-}
  echo $subid
  sbatch run_dimensionality_roi.sh $subid
done
