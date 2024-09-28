import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import numpy as np
import argparse
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision import transforms
def calculate_psnr(img_path, ground_truth_path):
    img = cv2.imread(img_path)
    ground_truth = cv2.imread(ground_truth_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    mse = np.mean((img - ground_truth) ** 2)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)+1e-10)
    return psnr


def calculate_ssim(img_path, ground_truth_path):
    img = cv2.imread(img_path)
    ground_truth = cv2.imread(ground_truth_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(img, ground_truth, data_range=ground_truth.max() - ground_truth.min())
    return ssim_value


def calculate_lpips(img_path, ground_truth_path):
    img = Image.open(img_path).convert('RGB')
    ground_truth = Image.open(ground_truth_path).convert('RGB')
 
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = preprocess(img).unsqueeze(0)
    ground_truth = preprocess(ground_truth).unsqueeze(0)
    loss_fn = lpips.LPIPS(net='vgg')
    lpips_value = loss_fn(img, ground_truth)
    return lpips_value.item()


def get_metrics(renders_dir, gt_dir):
    ssim_r=[]
    psnr_r=[]
    lpips_r=[]
    rend_imgs=os.listdir(renders_dir)
    gt_imgs=os.listdir(gt_dir)

    for fname in rend_imgs:
        render_path=os.path.join(renders_dir,fname)
        gt_path = os.path.join(gt_dir,fname)
        #--------------Assuming if groudtruth and renders are still in different order, then uncomment following-----------
        psnr_record=[]
        for gt_i in gt_imgs:
            gt_pt=os.path.join(gt_dir,gt_i)
            psnr_record.append(calculate_psnr(render_path, gt_pt))

        idx=psnr_record.index(max(psnr_record))
        gt_path=os.path.join(gt_dir,gt_imgs[idx])
        #-------------------------------------------------------------------------------------------------------------------
        ssim_r.append(calculate_ssim(render_path, gt_path))
        psnr_r.append(calculate_psnr(render_path, gt_path))
        lpips_r.append(calculate_lpips(render_path, gt_path))

    return ssim_r, psnr_r, lpips_r

def evaluate(target_dir, gt_dir):

    renders_dir=target_dir
    gt_dir=gt_dir

    ssims, psnrs, lpipss = get_metrics(renders_dir, gt_dir)
    print("```============================================='''")
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("```============================================='''\n")
    print("")
    text_file = open(f'{renders_dir}_metrics.txt', "w")
    text_file.write("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    text_file.write("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    text_file.write("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    text_file.close()


def main(gt_dir, target_dir):
    evaluate(target_dir, gt_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gt_dir',type=str,required=True,help='ground truth directory')
    parser.add_argument('--target_dir',type=str,required=True,help='target directory')
    args=parser.parse_args()
    main(args.gt_dir,args.target_dir)
