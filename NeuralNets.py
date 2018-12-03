import torch.nn.functional as F
import torch as tc
import numpy as np

class SimpleNet(tc.nn.Module):
    def __init__(self,dropout,features=3):
        super(SimpleNet,self).__init__()
        self.is_training=False
        self.conv1=tc.nn.Conv2d(features,8,kernel_size=(5,5))
        self.conv2=tc.nn.Conv2d(8,8,kernel_size=(3,3))
        self.drop=tc.nn.Dropout(dropout)
        # Net created for input_files 16*16
        self.fc1=tc.nn.Linear(100*8,256)
        self.fc2=tc.nn.Linear(256,1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,100*8)
        if self.is_training:
            x = self.drop(x)
        x = F.relu(self.fc1(x))
        if self.is_training:
            x = self.drop(x)
        x= tc.sigmoid(self.fc2(x))
        #x = self.fc2(x)
        return x