import matplotlib.image as mpimg
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from tqdm import tqdm
from scipy import ndimage

import torch.nn.functional as F
import torch as tc
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from helpers_img import *
from NeuralNets import *
from training_NN import *
from preprocessing import *

# Class to create the dataframe for U-Net 
class DatasetUNet(tc.utils.data.Dataset):
    # Constructor of the class
    def __init__(self, root_dir, bound=None, do_prep=False, do_flip = False,
                 normalize = False, noise=False, is_simple_noise=False, rot = False):
        ''' In this constructor you can decide to do or not:
            1. bound:
            2. do_prep: preprocessing
            3. do_flip: flip images
            4. normalize: normalize the tensor
            5. noise: add noise
            6. is_simple_noise:
            7. rot: rotate images'''
        
        self.image_dir = root_dir + "images/"
        self.files = os.listdir(self.image_dir)
        self.gt_dir = root_dir + "groundtruth/"
        self.do_prep = do_prep
        self.do_flip = do_flip
        self.normalize = normalize
        self.noise = noise
        self.is_simple_noise = is_simple_noise
        self.rot = rot
        if bound!=None:
            del self.files[0:bound[0]]
            del self.files[bound[1]:]
    
    def __len__(self):
        '''Return the '''
        return len(self.files)
    
    def true_len(self):
        '''Return the '''
        return len(self.files)*(1+3*self.rot)*(1+1*self.noise)*(1+1*self.do_flip)
    
    def __getitem__(self, index):
        '''????'''
        # Load images
        image = load_image(self.image_dir + self.files[index])
        gt_image = load_image(self.gt_dir + self.files[index])
        # Preprocess image
        if self.do_prep:
            _, laplacian_image = add_laplacian(image)
            sobel = add_sobel(image)
            segment = add_segment(image)
            image=np.concatenate((image,laplacian_image,sobel,segment),axis = 2)
        image,gt_image=[image],[gt_image]
        # Rotate images
        if self.rot:
            image,gt_image = rotation(image,gt_image, diagonal = True)
        # Flip images
        if self.do_flip:
            image,gt_image = flip(image,gt_image)
        
        # Image to tensor    
        image,gt_image = self.from_list_to_tensor(image), self.from_list_to_tensor(gt_image)
        
        # Normalize tensor
        if self.normalize:
            image = (image - image.mean())/image.std()
    
        # Add noise
        if self.noise:
            image,gt_image = self.add_noise(image, gt_image, is_simple=self.is_simple_noise)
        
        return image,gt_image
    
    def from_list_to_tensor(self,dataset):
        ''' cast a list of image in a tensor of appropriate size'''
        dataset = np.array(dataset)
        try :
            N,rows,columns,features = dataset.shape
        except:
            N,rows,columns = dataset.shape
            features=1
            dataset=dataset.reshape(N,rows,columns,features)

        dataset_tensor = tc.Tensor(N, features, rows, columns)
        for j in range(N):
            dataset_tensor[j] = tc.tensor(np.array([dataset[j,:,:,i] for i in range(features)]))


        return dataset_tensor
        
    def add_noise(self, dataset,label, is_simple=True):
        '''Add noise to the dataset.

        dataset : tensor type

        label : tensor type'''

        # If SIMPLE MEANSSSS
        if is_simple:

            mean, std = dataset.mean(), dataset.std()
            # The noise has the 20% of the image standard deviation
            noise = np.random.normal(loc = mean, scale = std/5, size = dataset.size())

            #
            dataset_with_noise = dataset + tc.tensor(noise).type(tc.FloatTensor)
            dataset = tc.cat((dataset,dataset_with_noise),dim = 0)
            label = label.type(tc.FloatTensor)
            label = tc.cat((label, label), dim = 0)
        else:

            mean, std = 0, 0.05

            noise = np.random.normal(mean, std, size = label.size())

            label.type(float)

            label_with_noise = label + tc.tensor(noise).type(tc.FloatTensor)

            label = tc.cat((label,label_with_noise),dim=0)

            dataset = tc.cat((dataset,dataset),dim=0)

        return dataset, label
    
    def get_features(self):
        ''' Returns the number of features'''
        if self.do_prep:
            features = 10
        else:
            features = 3
        return features
    def get_mini(self):
        '''REREERER'''
        return (1+3*self.rot)*(1+1*self.do_flip)*(1+1*self.noise)

# Class to create the dataset to test
class Testset(tc.utils.data.Dataset):
    # Constructor of the class
    def __init__(self, root_dir, nb_test_imgs, do_prep = False, normalize=False, expansion=False):
         ''' In this constructor you can decide to do or not:
             1. do_prep: preprocessing
             2. normalize: normalize the tensor
             3. expansion: reflect the border of image'''
        self.root_dir = root_dir
        self.nb_test_imgs = nb_test_imgs
        self.normalize = normalize
        self.do_prep = do_prep
        self.expansion = expansion
        
    def __getitem__(self,index):
        # Load image
        dir_test = self.root_dir + 'test_'+str(index+1)+'/'
        files_test = os.listdir(dir_test)
        img_test = load_image(dir_test + files_test[0])
        # Expand it
        if self.expansion:
            img_test = add_border(img_test,630)
        original_img = img_test
        # Preprocess it
        if self.do_prep:
            _, laplacian_image = add_laplacian(img_test)
            sobel = add_sobel(img_test)
            segment = add_segment(img_test)
            img_test=np.concatenate((img_test,laplacian_image,sobel,segment),axis = 2)
        img_test = self.from_list_to_tensor([img_test])
        # Normalize it
        if self.normalize:
            img_test = (img_test-img_test.mean())/img_test.std()
        return img_test, original_img
    
    def __len__(self):
        ''' return '''
        return self.nb_test_imgs
    
    def from_list_to_tensor(self,dataset):
        ''' cast a list of image in a tensor of appropriate size'''
        dataset = np.array(dataset)
        try :
            N,rows,columns,features = dataset.shape
        except:
            N,rows,columns = dataset.shape
            features=1
            dataset=dataset.reshape(N,rows,columns,features)

        dataset_tensor = tc.Tensor(N, features, rows, columns)
        for j in range(N):
            dataset_tensor[j] = tc.tensor(np.array([dataset[j,:,:,i] for i in range(features)]))


        return dataset_tensor
    
    def get_features(self):
        ''' Returns the number of features'''
        if self.do_prep:
            features = 10
        else:
            features = 3
        return features