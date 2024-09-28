import os
import argparse
import cv2
import numpy as np
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
import json
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips
from torchvision import transforms


parser=argparse.ArgumentParser()
parser.add_argument('--dir',type=str,required=True,help='directory of images to render upon')
parser.add_argument('--gt_dir',type=str,required=True, help='directory of ultimate ground truths')
args=parser.parse_args()
dirr=args.dir
gs_dir=dirr+'_gs'
if os.path.exists(gs_dir):
    os.system(f'rm -rf {gs_dir}')

os.system(f'mkdir {gs_dir}')

do_colmap=True
while do_colmap:
    if os.path.exists('dataset'):
        os.system('rm -rf dataset')

    os.system(f'ns-process-data images --data {dirr} --output-dir dataset --num-downscales 1 --sfm-tool hloc')
    if os.path.exists('dataset/images'):
        len1=len(os.listdir('dataset/images'))
        len2=len(os.listdir(dirr))
        if len1==len2:
            do_colmap=False


os.system(f'mkdir {gs_dir}/groundtruth_imgs')

gt_img_names=sorted(os.listdir('dataset/images'),key=lambda x:int(x.split('_')[-1].split('.')[0]))
gt_img_paths=[os.path.join('dataset/images',gt_name) for gt_name in gt_img_names]
gt_imgs=[cv2.imread(pth) for pth in gt_img_paths]  # destination
org_img_names=os.listdir(args.dir)
org_img_paths=[os.path.join(args.dir,img_name) for img_name in org_img_names]
org_imgs=[cv2.imread(pth) for pth in org_img_paths]  # source

# I am using assumption that org_imgs and pseduo_imgs has same indices in their filename in foramt _i

mapping_img_names={}
for i in range(len(gt_img_names)):
    img_gt=gt_imgs[i]
    errors=np.array([np.mean((img_gt-img_org)**2) for img_org in org_imgs])
    min_indx=np.argmin(errors)
    # mapping_img_names[int(org_img_names[min_indx].split('_')[-1].split('.')[0])]=int(gt_img_names[i].split('_')[-1].split('.')[0])-1
    mapping_img_names[int(gt_img_names[i].split('_')[-1].split('.')[0])-1]=int(org_img_names[min_indx].split('_')[-1].split('.')[0])

# now I need to create new_pseudo_gt_img_names from pseudo_get_img_names

pseudo_gt_img_names=sorted(os.listdir(args.gt_dir),key=lambda x:int(x.split('_')[-1].split('.')[0]) if '_' in x else int(x.split('.')[0]))
new_pseudo_names=[]
# for name in pseudo_gt_img_names:
#     print(name)
#     print("--------")
#     indx=int(name.split('_')[-1].split('.')[0])
#     new_indx=mapping_img_names[indx]
#     new_name=name.split('_')[0]+'_'+str(new_indx)+'.'+name.split('_')[-1].split('.')[-1]
#     new_pseudo_names.append(new_name)

for i in range(len(pseudo_gt_img_names)):
    name=pseudo_gt_img_names[i]
    new_idx=mapping_img_names[i]
    new_name=name.split('_')[0]+'_'+str(new_idx)+'.'+name.split('_')[-1].split('.')[-1]
    new_pseudo_names.append(new_name)
    
pseudo_gt_img_paths=[os.path.join(args.gt_dir,gt_name) for gt_name in new_pseudo_names]



os.system("ns-train splatfacto --data dataset  --viewer.quit-on-train-completion True --pipeline.model.cull_alpha_thresh 0.005 --pipeline.model.continue_cull_post_densification False nerfstudio-data --train-split-fraction 1.1 --eval_mode 'all'")

output_pth=os.listdir('outputs/dataset/splatfacto')[0]
config_pth=os.path.join(os.path.join('outputs/dataset/splatfacto',output_pth),'config.yml')
os.system(f'ns-render interpolate   --load-config {config_pth}   --pose-source eval   --frame-rate 100   --interpolation-steps 1   --output-path buffer.mp4')

buff_dirr='buffer.mp4'

cap = cv2.VideoCapture(buff_dirr)
os.system(f'mkdir {gs_dir}/rendered_imgs')
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")

count=0
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    if args.dir==args.gt_dir:
        gt_im=cv2.imread(gt_img_paths[count])
        cv2.imwrite(f'{gs_dir}/groundtruth_imgs/img_{count}.png',gt_im)
    else:
        tmp_pth=pseudo_gt_img_paths[count]
        print("++++++++++++")
        print(tmp_pth)
        gt_im=cv2.imread(tmp_pth)
        cv2.imwrite(f'{gs_dir}/groundtruth_imgs/img_{count}.png',gt_im)
    
    if frame.shape[0]==gt_im.shape[0] and frame.shape[1]==gt_im.shape[1]:
        cv2.imwrite(f'{gs_dir}/rendered_imgs/img_{count}.png',frame)
    else:
        frame=cv2.resize(frame,(gt_im.shape[0],gt_im.shape[1]))
        cv2.imwrite(f'{gs_dir}/rendered_imgs/img_{count}.png',frame)

    count+=1
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

os.system('rm -rf buffer.mp4')


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
    # Convert images to tensors
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

    for fname in gt_imgs:
        if '_' in fname:
            img_num=int(fname.split('_')[-1].split('.')[0])
        else:    
            img_num=int(fname.split('.')[0])
        if img_num%7==0:
            gt_path = os.path.join(gt_dir,fname)
            #--------------Assuming if groudtruth and renders are still in different order, then uncomment following-----------
            psnr_record=[]
            for rend_i in rend_imgs:
                rend_pt=os.path.join(renders_dir,rend_i)
                psnr_record.append(calculate_psnr(rend_pt, gt_path))
    
            idx=psnr_record.index(max(psnr_record))
            render_path=os.path.join(renders_dir,rend_imgs[idx])
            #-------------------------------------------------------------------------------------------------------------------
            ssim_r.append(calculate_ssim(render_path, gt_path))
            psnr_r.append(calculate_psnr(render_path, gt_path))
            lpips_r.append(calculate_lpips(render_path, gt_path))

    return ssim_r, psnr_r, lpips_r

def evaluate(base_dir):

    renders_dir=base_dir+'/rendered_imgs'
    gt_dir=base_dir+'/groundtruth_imgs'

    ssims, psnrs, lpipss = get_metrics(renders_dir, gt_dir)
    print("```============================================='''")
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    print("```============================================='''\n")
    print("")
    text_file = open(f'{base_dir}/metrics.txt', "w")
    text_file.write("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
    text_file.write("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
    text_file.write("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
    text_file.close()

evaluate(gs_dir)

os.system(f'ns-render interpolate   --load-config {config_pth}   --pose-source eval   --frame-rate 30   --interpolation-steps 30   --output-path {gs_dir}/rendering.mp4')
os.system('rm -rf outputs')