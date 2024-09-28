#!/bin/bash


python blurring.py --dir "$1"

echo "Motion Blurred"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_motion"
echo "Lens Blurred"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_lens"
echo "Gaussian Blurred"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_gaussian"

echo "Motion Deblurred (Real-ESRGAN)"
python deblur.py --dir "$1_blurred_motion"  --gt_dir "$1"
echo "Lens Deblurred (Real-ESRGAN)"
python deblur.py --dir "$1_blurred_lens" --gt_dir "$1"
echo "Gaussian Deblurred (Real-ESRGAN)"
python deblur.py --dir "$1_blurred_gaussian" --gt_dir "$1"
