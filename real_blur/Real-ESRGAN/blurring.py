import warnings
warnings.filterwarnings("ignore")

import cv2
import os
import numpy as np
import argparse
from blurgenerator import motion_blur,lens_blur, gaussian_blur

def calculate_psnr(img1, img2, max_value=255):
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)+1e-8))


def main(dirr,size,angle,radius,components,gamma):
    img_names=os.listdir(dirr)
    img_pths=[os.path.join(dirr,img_name) for img_name in img_names]
    new_dir1=dirr+'_blurred_motion'
    new_dir2=dirr+'_blurred_lens'
    new_dir3=dirr+'_blurred_gaussian'
    if os.path.exists(new_dir1):
        os.system(f'rm -rf {new_dir1}')
    os.makedirs(new_dir1)
    if os.path.exists(new_dir2):
        os.system(f'rm -rf {new_dir2}')
    os.makedirs(new_dir2)
    if os.path.exists(new_dir3):
        os.system(f'rm -rf {new_dir3}')
    os.makedirs(new_dir3)
    
    for i in range(len(img_pths)):
        img_name=img_names[i]
        pth=img_pths[i]
        image=cv2.imread(pth)
        motion_blurred = motion_blur(image, size=size, angle=angle)
        psnr=calculate_psnr(image,motion_blurred)
        print(f'Image:{img_name}')
        print(f'\t>>Motion Blurred Image has PSNR of {psnr} dB')
        cv2.imwrite(f'{new_dir1}/{img_name}',motion_blurred)
        lens_blurred=lens_blur(image,radius=radius,components=components,exposure_gamma=gamma)
        psnr=calculate_psnr(image,lens_blurred)
        cv2.imwrite(f'{new_dir2}/{img_name}',lens_blurred)
        print(f'\t>>Lens Blurred Image has PSNR of {psnr} dB')
        gaussian_blurred=gaussian_blur(image,size)
        psnr=calculate_psnr(image,gaussian_blurred)
        print(f'\t>>Gaussian Blurred Image has PSNR of {psnr} dB')
        cv2.imwrite(f'{new_dir3}/{img_name}',gaussian_blurred)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dir',type=str,required=True,help='directory of images')
    parser.add_argument('--size',type=int,default=120,help='size of blur kernel')
    parser.add_argument('--angle',type=int,default=30,help='direction of blur')
    parser.add_argument('--radius',type=int,default=10,help='radius of lens blur')
    parser.add_argument('--components',type=int,default=4,help='components of lens blur')
    parser.add_argument('--gamma',type=int,default=4,help='exposure gamma for lens blur')
    args=parser.parse_args()
    main(args.dir,args.size,args.angle,args.radius,args.components,args.gamma)




