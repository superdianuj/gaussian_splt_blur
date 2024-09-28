#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate torcher

cp -r "$1" NAFNet
cp -r "$1" Real-ESRGAN




echo "----------Motion Blur Zone-----------"


echo "NAFNet deblurring"
conda activate nafnet
cd NAFNet
python deblur.py --dir "$1"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_deblurred_nafnet" Real-ESRGAN


echo "Real-ESRGAN deblurring (also over NAFNet's output) - Motion Blurred"
conda activate torcher
cd Real-ESRGAN
python deblur.py --dir "$1"  --gt_dir "$1"
python deblur.py --dir "$1_deblurred_nafnet"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_deblurred_nafnet" .
cp -r "Real-ESRGAN/$1_deblurred_esrgan" .
cp -r "Real-ESRGAN/$1_deblurred_nafnet_deblurred_esrgan" .
conda activate nerfstudio
python gs_schedule.py --dir "$1_deblurred_nafnet" --gt_dir "$1"
python gs_schedule.py --dir "$1_deblurred_esrgan" --gt_dir "$1"
python gs_schedule.py --dir "$1_deblurred_nafnet_deblurred_esrgan" --gt_dir "$1"

