'''

Image Classification
10 kind of labels provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
Email: weason1998@gmail.com

Modified from torchvision, adapted to take 1*28*28
@function: Inference only
'''
from model.ResNet import ResNet18, ResNet34
from model.LeNet import LeNet
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import os
from dataset.MyDataset import CustomTensorDataset
from torch.utils.data import DataLoader
from PIL import Image

test_data_dir = r"./data/test_noise_rm.npy"
model_dir = r"./output/resnet_adpat-2019-12-01-lr_decay-0.03-bs-64-ep-100-folds-10-lrs-triangle-optim-AdamW-dist-False"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet_adpat"

# Number of classes in the dataset
num_classes = 10

# if you want to see what's in the training set
debug_img = 0

# Batch size for training (change depending on how much memory you have)
batch_size = 64

# folds
k_folds = 10


def set_parameter_requires_grad(model, feature_extracting):
    """
    When feature extract with pretrained model, we needn't retrain the parameters before FC
    But different when fine tune
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    # Other wise we will need to define the structure by ourselves with forward function using module and sequential to organize
    
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet52
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features # (fc): Linear(in_features=2048, out_features=1000, bias=True)
        model_ft.fc = nn.Linear(num_ftrs, num_classes) # replace fc with 2048 to num_class
        input_size = 224

    elif model_name == "LeNet":
        input_size = 28
        model_ft = LeNet()

    elif model_name == "resnet_adpat":
        input_size = 28
        model_ft = ResNet18(num_classes)

    elif model_name == "resnet_adpat34":
        input_size = 28
        model_ft = ResNet34(num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def inference(model, dataloader):
    print(' Begining testing')

    test_result = None
    for inputs, _ in dataloader:
        inputs = inputs.cuda()

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            if test_result is not None:
                test_result = torch.cat((test_result, outputs))
            else:
                test_result = outputs
    return test_result

if __name__ == "__main__":
    torch.cuda.set_device(2)

    model_ft, input_size = initialize_model(model_name, num_classes, False, use_pretrained=True)
    test_data = np.load(test_data_dir)
    test_mean = np.mean(test_data.ravel())
    test_std = np.std(test_data.ravel())
    test_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(input_size, interpolation=Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([test_mean/255.0], [test_std/255.0])])

    test_tensor_x = torch.stack([torch.Tensor(i) for i in test_data])
    test_tensor_x = test_tensor_x.reshape((-1, 1, input_size, input_size))

    test_dataset = CustomTensorDataset(tensors=(test_tensor_x, None), 
                                        transform=test_tf, 
                                        showing_img=False,
                                        is_training=False, 
                                        clone_to_three=False)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)
    all_test = None
    for fold in range(k_folds):
        model_path = os.path.join(model_dir, "models-fold-{}.pkl".format(int(fold)))
        model_ft.load_state_dict(torch.load(model_path))
        model_ft.cuda()
        model_ft.eval()
        test_result = inference(model_ft, test_dataloader)
        test_result = nn.functional.softmax(test_result, dim=1)
        if all_test is not None:
            all_test = torch.add(all_test, test_result)
        else:
            all_test = test_result
    _, preds = torch.max(all_test, 1)
    preds = preds.cpu().detach().numpy()
    csv_file = np.zeros((preds.shape[0],2), dtype=np.int32)
    csv_file[:,0] = np.arange(preds.shape[0])
    csv_file[:,1] = preds
    with open("./output/test-resnet_adpat_NoiseRM.csv", "wb") as f:
        f.write(b'image_id,label\n')
        np.savetxt(f, csv_file.astype(int), fmt='%i', delimiter=",")