import os
import cv2
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--dir',type=str,required=True)
args=parser.parse_args()


dirr=args.dir

new_dir=dirr+'_256'

if os.path.exists(new_dir):
    os.system(f'rm -rf {new_dir}')

os.system(f'mkdir {new_dir}')
        

dir_list=os.listdir(dirr)
im_paths=[os.path.join(dirr,curr_dir) for curr_dir in dir_list if curr_dir.endswith('png') or curr_dir.endswith('.jpg') or curr_dir.endswith('.JPG')]

counter=0
for pth in im_paths:
    img=cv2.imread(pth)
    new_pth=os.path.join(new_dir,str(counter)+'.png')
    img_resized=cv2.resize(img,(256,256))
    cv2.imwrite(new_pth,img_resized)
    counter+=1

# os.system(f'rm -rf {dirr}')
