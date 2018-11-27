
# coding: utf-8

# In[1]:


import os
import subprocess
import argparse
import numpy as np

parser = argparse.ArgumentParser()


# In[ ]:


def str2bool(v):
    return v.lower() in ('true')

parser.add_argument("--GOPRO_path", type = str, default = './GOPRO')
parser.add_argument("--output_path", type = str, default = './output')
parser.add_argument("--train_num", type = int, default = 1000)
parser.add_argument("--test_num", type = int, default = 10)
parser.add_argument("--is_gamma", type = str2bool, default = False)

args = parser.parse_args()

if not os.path.exists(os.path.join(args.output_path, 'train')):
    os.mkdir(os.path.join(args.output_path, 'train'))
    os.mkdir(os.path.join(args.output_path, 'train/sharp'))
    os.mkdir(os.path.join(args.output_path, 'train/blur'))
    
if not os.path.exists(os.path.join(args.output_path, 'test')):
    os.mkdir(os.path.join(args.output_path, 'test'))
    os.mkdir(os.path.join(args.output_path, 'test/sharp'))
    os.mkdir(os.path.join(args.output_path, 'test/blur'))

GOPRO_train_path = os.path.join(args.GOPRO_path, 'train')
GOPRO_test_path = os.path.join(args.GOPRO_path, 'test')

train_blur = []
train_sharp = []

for direc in sorted(os.listdir(GOPRO_train_path)):
    if args.is_gamma:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur_gamma')
    else:
        blur = os.path.join(os.path.join(GOPRO_train_path, direc), 'blur')    
    sharp = os.path.join(os.path.join(GOPRO_train_path, direc), 'sharp')
    
    sharp_imgs = sorted(os.listdir(sharp))
    for i, img in enumerate(sorted(os.listdir(blur))):
        train_blur.append(os.path.join(blur, img))
        train_sharp.append(os.path.join(sharp, sharp_imgs[i]))

train_blur = np.asarray(train_blur)
train_sharp = np.asarray(train_sharp)
random_index = np.random.permutation(len(train_blur))[:args.train_num]

for index in random_index:
    subprocess.call(['cp', train_blur[index], os.path.join(args.output_path, 'train/blur/%s'%('_'.join(train_blur[index].split('/')[-3:])))])
    subprocess.call(['cp', train_sharp[index], os.path.join(args.output_path, 'train/sharp/%s'%('_'.join(train_sharp[index].split('/')[-3:])))])


test_blur = []
test_sharp = []

for direc in sorted(os.listdir(GOPRO_test_path)):
    if args.is_gamma:
        blur = os.path.join(os.path.join(GOPRO_test_path, direc), 'blur_gamma')
    else:
        blur = os.path.join(os.path.join(GOPRO_test_path, direc), 'blur')    
    sharp = os.path.join(os.path.join(GOPRO_test_path, direc), 'sharp')
    
    sharp_imgs = sorted(os.listdir(sharp))
    for i, img in enumerate(sorted(os.listdir(blur))):
        test_blur.append(os.path.join(blur, img))
        test_sharp.append(os.path.join(sharp, sharp_imgs[i]))

test_blur = np.asarray(test_blur)
test_sharp = np.asarray(test_sharp)
random_index = np.random.permutation(len(test_blur))[:args.test_num]

for index in random_index:
    subprocess.call(['cp', test_blur[index], os.path.join(args.output_path, 'test/blur/%s'%('_'.join(test_blur[index].split('/')[-3:])))])
    subprocess.call(['cp', test_sharp[index], os.path.join(args.output_path, 'test/sharp/%s'%('_'.join(test_sharp[index].split('/')[-3:])))])
    
    

