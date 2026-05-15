#!/bin/bash

for subpath in /bgfs/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do 
  fullsubid=$(basename $subpath)
  subid=${fullsubid#sub-}

# FLT06 FLT08 FLT11 FLT14 FLT18 FLT26 
#for subid in FLT02 FLT03 FLT04 FLT05 FLT07 FLT09 FLT10 FLT12 FLT13 FLT15 FLT17 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 #FLT28 FLT30; do
  echo $subid
  sbatch run_modeling_firstlevel_GLMsingle_stgrid.sh $subid
done
