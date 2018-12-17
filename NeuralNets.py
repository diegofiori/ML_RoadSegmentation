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
    
class UNet(tc.nn.Module):
    def __init__(self,features=3):
        super(UNet,self).__init__()
        self.is_training=False
        self.conv1 = tc.nn.Conv2d(features,8,kernel_size=3)
        self.batch1 = tc.nn.BatchNorm2d(8)
        self.conv2 = tc.nn.Conv2d(8,8, kernel_size=3)
        self.batch2 = tc.nn.BatchNorm2d(8)
        self.conv3 = tc.nn.Conv2d(8,16, kernel_size=3)
        self.batch3 = tc.nn.BatchNorm2d(16)
        self.conv4 = tc.nn.Conv2d(16,16, kernel_size=3)
        self.batch4 = tc.nn.BatchNorm2d(16)
        self.conv5 = tc.nn.Conv2d(16,16, kernel_size=4)
        self.batch5 = tc.nn.BatchNorm2d(16)
        self.convUp1 = tc.nn.ConvTranspose2d(16,16,stride=2,kernel_size=2)
        self.conv6 = tc.nn.Conv2d(32,16, kernel_size=3)
        self.batch6 = tc.nn.BatchNorm2d(16)
        self.conv7 = tc.nn.Conv2d(16,16, kernel_size=3)
        self.batch7 = tc.nn.BatchNorm2d(16)
        self.convUp2 = tc.nn.ConvTranspose2d(16,8,stride=2,kernel_size=2)
        self.conv8 = tc.nn.Conv2d(16,8, kernel_size=3)
        self.batch8 = tc.nn.BatchNorm2d(8)
        self.conv9 = tc.nn.Conv2d(8,8, kernel_size=3)
        self.batch9 = tc.nn.BatchNorm2d(8)
        self.convUp3 = tc.nn.ConvTranspose2d(8,4,kernel_size=39)
        self.conv10 = tc.nn.Conv2d(4,1,kernel_size=3)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.batch1(x)
        x = F.relu(self.conv2(x))
        #x1 = x.clone()
        x1 = x
        x = F.max_pool2d(x, kernel_size=2)
        x = x = self.batch2(x)
        x = F.relu(self.conv3(x))
        x = self.batch3(x)
        x = F.relu(self.conv4(x))
        #x2 = x.clone()
        x2 = x
        x = F.max_pool2d(x, kernel_size=2)
        x = self.batch4(x)
        x = F.relu(self.conv5(x))
        x = self.batch5(x)
        
        x = F.relu(self.convUp1(x))
        
        x= self.adapt_activation(x2,x)
        
        x = F.relu(self.conv6(x))
        x = self.batch6(x)
        x = F.relu(self.conv7(x))
        x = self.batch7(x)
        
        x = F.relu(self.convUp2(x))
        
        x = self.adapt_activation(x1,x)
        
        x = F.relu(self.conv8(x))
        x = self.batch8(x)
        x = F.relu(self.conv9(x))
        x = self.batch9(x)
        
        x = F.relu(self.convUp3(x))
        x = tc.sigmoid(self.conv10(x))
        
        return x
    
    def adapt_activation(self,x_old,x_new):
        '''the function adapts the activation to the smaller one dimensions
        and concatenates them'''
        
        space = (x_old.size(2)- x_new.size(2))//2
        
        distance = x_new.size(2)
        
        x_old = x_old.narrow(2,space,distance)
        x_old = x_old.narrow(3,space,distance)
        
        x_new = tc.cat((x_old,x_new),dim=1)
        
        return x_new