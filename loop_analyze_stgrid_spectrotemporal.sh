#!/bin/bash

for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do
  fullsubid=$(basename $subpath)
  subid=${fullsubid#sub-}

  echo $subid
  sbatch run_analyze_stgrid_spectrotemporal.sh $subid
done
