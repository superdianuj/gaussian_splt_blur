#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

python blurring.py --dir "$1"

echo "---Motion Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_motion"
echo "---Lens Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_lens"
echo "---Gaussian Blurred---"
python compute_metrics.py --gt_dir "$1" --target_dir "$1_blurred_gaussian"
