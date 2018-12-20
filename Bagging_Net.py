%matplotlib inline
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
from NeuralNets import *
from training_NN import *
from Post_processing import *

def bootstrap_images(imgs, gt_imgs,size,number):
    '''This function returns a list of list. Each element of the "external" list is a list of randomly 
    sampled images with replacement'''
    
    new_imgs=[]
    new_gt_imgs=[]
    array = np.arange(len(imgs))
    matrix= np.zeros((len(imgs),number))
    for k in range(number):
        b = np.random.choice(array, size, replace=True)
        list_temp_imgs = [imgs[i] for i in b]
        list_temp_gt_imgs = [gt_imgs[i] for i in b]
        new_imgs.append(list_temp_imgs)
        new_gt_imgs.append(list_temp_gt_imgs)
        matrix[b,k]=1
    
    return new_imgs,new_gt_imgs,matrix


def train_SimpleNet(dataset, label, w, h, lr, max_epochs, mini_batch_size, dropout):
    ''' Train a simple net'''
    n = len(dataset)
    train_sub_images = [img_crop(dataset[i], w, h) for i in range(n)]
    train_mask_label = [img_crop(label[i],w,h) for i in range(n)]
    train_mask_label = from_mask_to_vector(train_mask_label,0.3)
    train_sub_images = transform_subIMG_to_Tensor(train_sub_images)
    mean = train_sub_images.mean()
    std = train_sub_images.std()
    train_sub_images = (train_sub_images-mean)/std
    train_sub_images, train_mask_label = reduce_dataset(train_sub_images,train_mask_label)
    # shuffle images
    for l in range(10):
        new_indices= np.random.permutation(len(train_mask_label))
        train_sub_images=train_sub_images[new_indices]
        train_mask_label=train_mask_label[new_indices]
    
    model = SimpleNet(dropout)
    
    mini_batch_rest = train_sub_images.size(0) % mini_batch_size
    
    if mini_batch_rest > 0:
        train_sub_images = train_sub_images.narrow(0,0,train_sub_images.size(0)-mini_batch_rest)
        train_mask_label = train_mask_label[0:train_sub_images.size(0)]
        
        
        

    train_model_Adam( model, train_sub_images, train_mask_label, max_epochs, lr, mini_batch_size)
    
    return model

def bagging_NN(dataset, label, percentage_train_data, nb_model, w, h, lr, max_epochs, mini_batch_size, dropout):
    nb_data = int( len(dataset)*percentage_train_data )
    list_dataset, list_label, data_matrix = bootstrap_images(dataset, label, nb_data, nb_model)
    models = []
    for i in range(nb_model):
        model=train_SimpleNet(list_dataset[i], list_label[i], w, h, lr, max_epochs, mini_batch_size, dropout)
        models.append(model)
        print('model '+str(i)+' trained')
        
    data_matrix = 1 - data_matrix
    # the data matrix has 1 in position n,j if the nth image was not used in jth
    # model training.
    
    # compute F1 error
    
    test_imgs=[img_crop(dataset[k], w, h) for k in range(len(dataset))]
    nb_patches=len(test_imgs[0])
    test_imgs = transform_subIMG_to_Tensor(test_imgs)
    mean=test_imgs.mean()
    std= test_imgs.std()
    test_imgs = (test_imgs-mean)/std
    F1_error=0
    not_testable_img=0
    for i in range(len(dataset)):
        image= test_imgs.narrow(0,i*nb_patches,nb_patches)
        if data_matrix[i,:].sum()>0:
            ind=np.where(data_matrix[i,:])[0]
            predictions=[models[k](image).detach().numpy() for k in ind]
            predictions = np.array(predictions)
            predictions = ((predictions.mean(0)[:] >0.5)*1).reshape(-1,)
            mask_test = label_to_img(400, 400, w, h, predictions)
            F1_error += calcul_F1(label[i], mask_test)
        else:
            not_testable_img+=1
    
    F1_error= F1_error/(len(dataset)-not_testable_img)
    return models, F1_error


def train_model_Adam( model, train_data, label, max_epochs, lr, mini_batch_size, threshold=0.01):
    '''train the Neural Net using Adam as optimizer and an MSE loss'''
    optimizer=tc.optim.Adam(model.parameters(),lr)
    criterion= tc.nn.MSELoss()
    training_errors=[]
    if tc.cuda.is_available():
        tc.cuda.empty_cache()
        model.cuda()
        train_data = train_data.cuda()
    
    for epoch in tqdm(range(max_epochs)):
        model.is_training=True
        model.train()
        if tc.cuda.is_available():
            tc.cuda.empty_cache()
        for i in range(0,train_data.size(0),mini_batch_size):
            output= model(train_data.narrow(0,i,mini_batch_size))
            temp=tc.FloatTensor(np.array([1*label[i:i+mini_batch_size]]).reshape(-1,1))
            
            temp = temp.cuda()
            loss= criterion(output,temp)
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
