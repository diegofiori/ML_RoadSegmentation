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


