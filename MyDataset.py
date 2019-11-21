import torch
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None, is_training=True, clone_to_three=False):
        self.train = is_training
        if self.train:
            assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.clone_to_three = clone_to_three
        

    def __getitem__(self, index):
        x = self.tensors[0][index]

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
