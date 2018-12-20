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
from dataset import *

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
    
def train_UNet(training_directory, lr, max_epochs, mini_batch_size, nb_test, threshold=0.5, 
               do_preprocessing = False, flip_data=True, model=None, model_path = 'Model_UNet/model_CPU.pt'):
    ''' train the UNet using the Binary Cross Entropy loss and Adam as optimizer.
    The dataset must be a list of 400*400 images.'''
    
    dataset = DatasetUNet(training_directory, bound=(0,-nb_test) ,do_flip=flip_data,
                          do_prep= do_preprocessing,
                          noise=True, is_simple_noise = True, rot = True, normalize = True)
    N = dataset.__len__()+ nb_test
    test_set = DatasetUNet(training_directory, bound=(N-nb_test,N) ,do_flip=False,
                           do_prep= do_preprocessing,
                           noise=False, is_simple_noise = False, rot = False, normalize = True)
    train_load = tc.utils.data.DataLoader(dataset,batch_size= mini_batch_size)
    test_load = tc.utils.data.DataLoader(test_set,batch_size=nb_test)
    if model == None:
        model = UNet(features= dataset.get_features())
    
    optimizer=tc.optim.Adam(model.parameters(),lr)
    # maybe using MSE is better
    #criterion= tc.nn.BCELoss()
    criterion = tc.nn.MSELoss()
    training_errors=[]
    losses=[]
    
    if tc.cuda.is_available():
        print('cuda is available')
        model.cuda()
        #criterion.cuda()
        #dataset= dataset.cuda()
        #label = label.cuda()
    
    #training_F1_error=[]
    #print('starting to train the net')
    for epoch in tqdm(range(max_epochs)):
        model.train()
        #if tc.cuda.is_available():
        #    model.cuda()
        for input_data, label_data in train_load:
            input_data = input_data.view(dataset.get_mini()*mini_batch_size,dataset.get_features(),400,400)
            label_data = label_data.view(dataset.get_mini()*mini_batch_size,1,400,400)
            if tc.cuda.is_available():
                input_data, label_data = input_data.cuda(), label_data.cuda()
            output = model(input_data)
            #print(output, label_data)
            loss = criterion(output, label_data)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            tc.cuda.empty_cache()
            
        
        # compute training error
        model.eval()
        losses.append(loss)
        for test,mask in test_load:
            test = test.view(test_set.get_mini()*nb_test,test_set.get_features(),400,400)
            mask = mask.view(test_set.get_mini()*nb_test,1,400,400)
            if tc.cuda.is_available():
                test = test.cuda()
            prediction = model(test)
            prediction = prediction.cpu()
            prediction = prediction.detach_().numpy()[:,0,:,:]
            prediction = (prediction > threshold)*1
            #print(prediction[0])
            mask = mask.numpy()[:,0,:,:]
            F1_error = 0
            #print(mask.shape)
            training_error = (((mask>0.5)*1 == prediction)*1).sum()/np.prod(prediction.shape)

            training_errors.append(training_error)
        
    model.cpu()
    tc.save(model,model_path)
   
    plt.figure()
    plt.plot(np.arange(epoch + 1)+1,losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    try:
        plt.figure()
        plt.plot(np.arange(epoch + 1)+1,training_errors)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
    except:
        print(training_errors)
    return model
def train_model_Adam_v2( model, dataset, max_epochs, lr, mini_batch_size, w=48, h=48, features=3, threshold=0.01):
    '''train the Neural Net using Adam as optimizer and an binary cross entropy loss.
    The function is written explicitly for the DeepNet.'''
    
    train_loader = DataLoader(dataset,batch_size=mini_batch_size)
    optimizer=tc.optim.Adam(model.parameters(),lr)
    criterion= tc.nn.BCELoss()
    losses=[]
    training_errors = []
    if tc.cuda.is_available():
        model.cuda()
        criterion.cuda()
    
    for epoch in tqdm(range(max_epochs)):
        model.is_training=True
        model.train()
    
        for train_data,label in train_loader:
            train_data = train_data.view(-1,features,w,h)
            label = label.view(-1,1).type(tc.FloatTensor)
            if tc.cuda.is_available():
                train_data = train_data.cuda()
                label = label.cuda()
            output= model(train_data).view(-1,1)
            #print(output,tc.LongTensor(np.array([1*label[i:i+mini_batch_size]]).reshape(-1,1)))
            loss= criterion(output,label)
            model.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss)
        
    plt.figure()
    plt.plot(np.arange(epoch+1)+1,losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
    model.cpu()    
    return model

def trainDeepNet(root_dir, max_epochs, lr, mini_batch_size, dropout=0, model = None):
    if model == None:
        model = DeepNet(dropout)
    
    dataset = DatasetDeepNet(root_dir, do_flip=True, do_rotation=True,do_train=False)
    
    model = train_model_Adam_v2( model, dataset, max_epochs, lr, mini_batch_size)
    return model