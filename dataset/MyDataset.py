'''

Image Classification
10 kind of labels provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
Email: weason1998@gmail.com

@function: Dataset for npy-format img input
'''

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, is_training=True, showing_img=False, clone_to_three=False):
        self.train = is_training
        if self.train:
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.clone_to_three = clone_to_three
        self.show = showing_img

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.show:
            CustomTensorDataset.imshow(x) #torchvision.utils.make_grid(x, 4))

        if self.transform:
            x = self.transform(x)
        
        if self.clone_to_three:
            # Adapt to 3 channel        
            x = torch.cat([x,x,x], dim=0)

        if self.train:
            y = self.tensors[1][index]
        else:
            y = torch.zeros((1,))
            
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

    @staticmethod
    def imshow(img, title=''):
        """Plot the one-channel image batch from the tensor
        """
        plt.figure(figsize=(10, 10))
        plt.title(title)
        plt.imshow(np.squeeze(np.transpose(img.numpy(), (1, 2, 0))), cmap='gray')
        plt.show()

