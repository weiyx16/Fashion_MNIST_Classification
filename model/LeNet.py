'''

Image Classification
10 kind of labels provided from assistant teachers
Pytorch 1.1.0 & python 3.6

Author: @weiyx16.github.io
Email: weason1998@gmail.com

@function: LeNet
'''
import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True),  
            #padding=2，28+2*2 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 ,stride = 2)
        )
 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),   
            #input_size=(6*14*14)，output_size=16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2,stride = 2)
        )
 
        self.fc1 = nn.Sequential(
            nn.Linear(16*5*5,120),
            nn.ReLU()
        )
 
        self.fc2 = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU()
        )
 
        self.fc3 = nn.Linear(84,10)
 
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
 
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
