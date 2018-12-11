import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import torch.nn.functional as F
import torch as tc
from helpers_img import *

def train_model_Adam( model, train_data, label, max_epochs, lr, mini_batch_size, threshold=0.01):
    '''train the Neural Net using Adam as optimizer and an MSE loss'''
    #optimizer=tc.optim.SGD(model.parameters(),lr)
    optimizer=tc.optim.Adam(model.parameters(),lr)
    criterion= tc.nn.MSELoss()
    training_errors=[]
    if tc.cuda.is_available():
        model.cuda()
        train_data = train_data.cuda()
    
    for epoch in tqdm(range(max_epochs)):
        model.is_training=True
        model.train()
        if tc.cuda.is_available():
            tc.cuda.empty_cache()
        for i in range(0,train_data.size(0),mini_batch_size):
            output= model(train_data.narrow(0,i,mini_batch_size))
            #print(output,tc.LongTensor(np.array([1*label[i:i+mini_batch_size]]).reshape(-1,1)))
            loss= criterion(output,tc.FloatTensor(np.array([1*label[i:i+mini_batch_size]]).reshape(-1,1)))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        # compute training error
        model.is_training=False
        model.eval()
        test = model(train_data)
        test = test.cpu()
        prediction= test[:]>0.5
        
        prediction= 1*(prediction.numpy()[:] != label.reshape(-1,1)[:])
        
        training_error = np.sum(prediction)/len(prediction)
        training_errors.append(training_error*100)
        if training_error< threshold:
            break
        
    
    plt.figure()
    plt.plot(np.arange(epoch+1)+1,training_errors)
    plt.xlabel('epoch')
    plt.ylabel('error [%]')
    plt.show()
        
    model.cpu()    
    