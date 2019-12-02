import numpy as np
import os
from PIL import Image
import cv2

npy_path = r'./data/test.npy'
save_path = r'./data/test_noise_rm.npy'

data = np.load(npy_path)
data = np.reshape(data, (data.shape[0], 28, 28))
data_noise_rm = np.zeros(data.shape)

for i in range(data.shape[0]):
    data_save = cv2.medianBlur(data[i], 3)
    data_noise_rm[i,:,:] = data_save
   
data_noise_rm = np.reshape(data_noise_rm, (data.shape[0], 784))
np.save(save_path, data_noise_rm)
