#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate torcher

cp -r "$1" synthetic_blurring
cp -r "$1" NAFNet
cp -r "$1" Real-ESRGAN

cd synthetic_blurring
python blurring.py --dir "$1"

echo "---Motion Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_motion"
echo "---Lens Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_lens"
echo "---Gaussian Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_gaussian"
cd ..




echo "----------Motion Blur Zone-----------"
cp -r "synthetic_blurring/$1_blurred_motion" NAFNet
cp -r "synthetic_blurring/$1_blurred_motion" Real-ESRGAN


echo "NAFNet deblurring - Motion Blurred"
conda activate nafnet
cd NAFNet
python deblur.py --dir "$1_blurred_motion"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_motion_deblurred_nafnet" Real-ESRGAN


echo "Real-ESRGAN deblurring (also over NAFNet's output) - Motion Blurred"
conda activate torcher
cd Real-ESRGAN
python deblur.py --dir "$1_blurred_motion"  --gt_dir "$1"
python deblur.py --dir "$1_blurred_motion_deblurred_nafnet"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_motion_deblurred_nafnet" .
cp -r "Real-ESRGAN/$1_blurred_motion_deblurred_esrgan" .
cp -r "Real-ESRGAN/$1_blurred_motion_deblurred_nafnet_deblurred_esrgan" .
conda activate nerfstudio
python gs_schedule.py --dir "$1_blurred_motion_deblurred_nafnet" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_motion_deblurred_esrgan" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_motion_deblurred_nafnet_deblurred_esrgan" --gt_dir "$1"




echo "----------Gaussian Blur Zone-----------"
cp -r "synthetic_blurring/$1_blurred_gaussian" NAFNet
cp -r "synthetic_blurring/$1_blurred_gaussian" Real-ESRGAN


echo "NAFNet deblurring - Gaussian Blurred"
conda activate nafnet
cd NAFNet
python deblur.py --dir "$1_blurred_gaussian"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_gaussian_deblurred_nafnet" Real-ESRGAN


echo "Real-ESRGAN deblurring (also over NAFNet's output) - Gaussian Blurred"
conda activate torcher
cd Real-ESRGAN
python deblur.py --dir "$1_blurred_gaussian"  --gt_dir "$1"
python deblur.py --dir "$1_blurred_gaussian_deblurred_nafnet"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_gaussian_deblurred_nafnet" .
cp -r "Real-ESRGAN/$1_blurred_gaussian_deblurred_esrgan" .
cp -r "Real-ESRGAN/$1_blurred_gaussian_deblurred_nafnet_deblurred_esrgan" .
conda activate nerfstudio
python gs_schedule.py --dir "$1_blurred_gaussian_deblurred_nafnet" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_gaussian_deblurred_esrgan" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_gaussian_deblurred_nafnet_deblurred_esrgan" --gt_dir "$1"








echo "----------Lens Blur Zone-----------"
cp -r "synthetic_blurring/$1_blurred_lens" NAFNet
cp -r "synthetic_blurring/$1_blurred_lens" Real-ESRGAN


echo "NAFNet deblurring - Lens Blurred"
conda activate nafnet
cd NAFNet
python deblur.py --dir "$1_blurred_lens"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_lens_deblurred_nafnet" Real-ESRGAN


echo "Real-ESRGAN deblurring (also over NAFNet's output) - Lens Blurred"
conda activate torcher
cd Real-ESRGAN
python deblur.py --dir "$1_blurred_lens"  --gt_dir "$1"
python deblur.py --dir "$1_blurred_lens_deblurred_nafnet"  --gt_dir "$1"
cd .. 
cp -r "NAFNet/$1_blurred_lens_deblurred_nafnet" .
cp -r "Real-ESRGAN/$1_blurred_lens_deblurred_esrgan" .
cp -r "Real-ESRGAN/$1_blurred_lens_deblurred_nafnet_deblurred_esrgan" .
conda activate nerfstudio
python gs_schedule.py --dir "$1_blurred_lens_deblurred_nafnet" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_lens_deblurred_esrgan" --gt_dir "$1"
python gs_schedule.py --dir "$1_blurred_lens_deblurred_nafnet_deblurred_esrgan" --gt_dir "$1"


