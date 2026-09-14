#!/bin/bash

#for subpath in /ix1/bchandrasekaran/krs228/data/FLT/data_denoised/sub*/; do
#  fullsubid=$(basename $subpath)
#  subid=${fullsubid#sub-}

# fMRIPrep 25.2.5 / native T1w recompute (09_integration/INTEGRATION_PLAN.md, Workstream I) --
# subject list is the 12 non-Mandarin subjects analyzed in the manuscript
# (fMRI_auditory-category-learning/REVISION_PLAN.md).
#
# FLAG: the previous list here (kept below for reference) excluded FLT06, FLT08, FLT11,
# FLT14, FLT18, FLT26 from this repo's GLMsingle masking pipeline for an undocumented reason.
# FLT06/FLT11/FLT14 ARE part of the manuscript's 12-subject non-Mandarin sample, so they're
# added back in here -- but confirm their 25.2.5 fMRIPrep QA and GLMsingle fit are actually
# clean before trusting their dimensionality output. If the original exclusion reason still
# holds, drop them back out and treat analysis B as an n<12 comparison for those ROIs.
# Previous list (18 subjects, Mandarin + non-Mandarin, FLT06/11/14 excluded):
#   FLT02 FLT03 FLT04 FLT05 FLT07 FLT09 FLT10 FLT12 FLT13 FLT15 FLT17 FLT19 FLT20 FLT21 FLT22 FLT23 FLT24 FLT25 FLT28 FLT30
for subid in FLT02 FLT04 FLT06 FLT09 FLT11 FLT12 FLT13 FLT14 FLT20 FLT25 FLT28 FLT30; do
  echo $subid
  sbatch run_mask_glmsingle_betas.sh $subid
done
