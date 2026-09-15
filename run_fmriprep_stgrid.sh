#!/bin/bash
#SBATCH --time=3-00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Preprocess single-subject FLT data using fmriprep
# in a Singularity container
# Updated to fmriprep 25.2.5 (current LTS line, support through Oct 2029)

module add freesurfer
module add fsl
module add afni
module add ants
module add singularity

#conda activate py3

# define paths
software_path=/ix1/bchandrasekaran/krs228/software/
project_path=/ix1/bchandrasekaran/krs228/data/FLT/
data_dir=$project_path/data_denoised/

fmriprep_version=25.2.5
analysis_desc="denoised_fmriprep-$fmriprep_version"
work_dir=/ix1/bchandrasekaran/krs228/work/${analysis_desc}
out_dir=$data_dir/derivatives/${analysis_desc}/

# singularity
sing_dir=$software_path/singularity_images/
sing_img=$sing_dir/fmriprep-${fmriprep_version}.simg

# define inputs
fs_license=$software_path/license.txt
sub=$1

# NOTE: intentionally NOT reusing the 22.1.1-era FreeSurfer outputs here.
# The bundled FreeSurfer version has moved on since 22.1.1 (7.2 -> 7.3.2+),
# and mixing FreeSurfer versions across a derivatives tree risks subtle
# inconsistencies. Omitting --fs-subjects-dir lets fmriprep run recon-all
# fresh under this run's own <out_dir>/sourcedata/freesurfer/ (the
# --output-layout bids default). This adds several hours/subject of
# recon-all compute versus the old script -- budget for it.

# copy from SBATCH arguments
mem=64000
nprocs=8
omp_n=4

# BEFORE RUNNING FOR THE FIRST TIME:
# build the fmriprep container to a singularity image
# (will only build from head node; no unsquashfs when running from nodes)
#singularity build $sing_img docker://nipreps/fmriprep:25.2.5

# run fmriprep
singularity run --cleanenv -B /ix1:/ix1 $sing_img \
  $data_dir $out_dir participant \
  --participant-label $sub \
  --task stgrid \
  --fs-license-file $fs_license \
  --work-dir $work_dir \
  --skip-bids-validation \
  --output-layout bids \
  -vv \
  --mem $mem \
  --nprocs $nprocs --omp-nthreads $omp_n \
  --output-spaces T1w fsnative MNI152NLin2009cAsym
