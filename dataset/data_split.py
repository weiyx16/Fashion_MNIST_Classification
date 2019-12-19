'''

Image Classification with finetuning or feature extraction on pretrained resnet-50
10 kind of labels provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
weiyx16@mails.tsinghua.edu.cn

adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

@function: Split src training set with 30K img to training set (80%) and validation set (80%)
'''

import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    src_data_path = r"../data/src.npy"
    src_gt_path = r"../data/src.csv"
    data = np.load(src_data_path)  # (30000, 784)
    mean = np.mean(data.ravel())
    std = np.std(data.ravel())

    gt = pd.read_csv(src_gt_path)

    data_len = data.shape[0]
    split_partial = 0.15
    shuffle_idx = np.arange(data_len)
    np.random.shuffle(shuffle_idx)
    val_set_end = round(data_len * split_partial ,0)
    val_idx = shuffle_idx[:int(val_set_end)]
    train_idx = shuffle_idx[int(val_set_end):]

    # split the img data
    val_data = data[val_idx,:]
    assert val_data.shape[1] == data.shape[1], "Error during splitting validation set"
    train_data = data[train_idx,:]
    assert train_data.shape[1] == data.shape[1], "Error during splitting training set"
    if not os.path.exists("../data"):
        os.mkdir("../data")
    np.save("../data/validation.npy", val_data)
    np.save("../data/train.npy", train_data)

    # split the gt
    val_gt = gt.iloc[val_idx, :]
    train_gt = gt.iloc[train_idx,:]
    # set new idx to each file
    val_gt.image_id=np.arange(len(val_data))
    train_gt.image_id=np.arange(len(train_data))
    
    val_gt.to_csv("../data/validation.csv", index=False)
    train_gt.to_csv("../data/train.csv", index=False)

    print(" >> Split the src dataset into {} training img, and {} validation img" .format(train_data.shape[0], val_data.shape[0]))
    print(" Transforms.Normalize(mean = {}, std = {})" .format(mean, std))