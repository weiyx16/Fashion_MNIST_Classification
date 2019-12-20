'''

Image Classification
10 kind of labels provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
Email: weason1998@gmail.com

@function: Convert the img from npy to png and see what's on earth
'''
import numpy as np
import os
from PIL import Image
import cv2

npy_path = r'../data/test.npy'
gt_path = None #r'./data/train.csv'
save_dir = r'../data/test'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

data = np.load(npy_path)
data = np.reshape(data, (data.shape[0], 28, 28))
if gt_path is not None:
    gt = np.genfromtxt(gt_path, delimiter=',')[1:,]

for i in range(data.shape[0]):
    data_save = data[i]
    img = Image.fromarray(data_save)
    if gt_path is not None:
        img.save(os.path.join(save_dir,'{:5}_{}.png'.format(i, int(gt[i,1]))))
    else:
        img.save(os.path.join(save_dir,'{:5}.png'.format(i)))