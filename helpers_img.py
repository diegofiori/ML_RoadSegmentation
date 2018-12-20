import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import torch.nn.functional as F
import torch as tc
from mask_to_submission import *



# Helper functions

def load_image(infilename):
    """
    Use the scipy.misc to load the image.
    """
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img):
    """
    Change the image elements type.
    """
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg

def concatenate_images(img, gt_img):
    """
    Concatenate an image and its groundtruth.
    """
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def value_to_class(v):
    """
    Assign a class 1/0 (road or background) to v.
    """
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def img_crop(im, w, h):
    """
    Crops an image im in patches of size w,h.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def img_crop_mod(im, w, h):
    """
    Crops an image im in patches of size 3*w,3*h.
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(h,imgheight-h,h):
        for j in range(w,imgwidth-w,w):
            if is_2d:
                im_patch = im[j-w:j+2*w, i-h:i+2*h]
            else:
                im_patch = im[j-w:j+2*w, i-h:i+2*h, :]
            list_patches.append(im_patch)
    return list_patches

def from_mask_to_vector(mask_imgs,threshold):
    """
    Takes as input a list of mask and returns a vector where the ith component
    is 1 if the majority of the ith mask list element is 1.
    """
    mask_imgs=np.array(mask_imgs).reshape(-1,16,16)
    vector=mask_imgs.sum(axis=(1,2))
    vector=vector[:] > mask_imgs.shape[1]*mask_imgs.shape[2]*threshold
    return vector

def transform_subIMG_to_Tensor(sub_img_list):
    '''Transform a list of list of sub images into a Tensor,
    where the first index point to different sub images'''
    N0= len(sub_img_list)
    N1= len(sub_img_list[0])
    N=N0*N1
    x,y,channels=sub_img_list[0][0].shape
    tensor=tc.Tensor(N,channels,x,y)
    for j,image in enumerate(sub_img_list):
        for k,sub_image in enumerate(image):
            tensor[j*N1+k,:,:,:]=tc.FloatTensor(np.array([sub_image[:,:,i] for i in range(sub_image.shape[2])]))
    
    return tensor


def reduce_dataset(dataset,label):
    """
    Balance the number of zeros and ones in an image.
    """
    new_dataset=[]
    new_label=[]
    nb_zeros=0
    ones_positions=np.where(label)[0]
    for pos in ones_positions:
        new_dataset.append(dataset.narrow(0,int(pos),1))
        new_label.append(1)
        
    while nb_zeros < len(ones_positions):
        j= np.random.randint(0,dataset.size(0))
        if label[j]==0:
            #print(j)
            new_dataset.append(dataset.narrow(0,j,1))
            new_label.append(0)
            nb_zeros+=1
    new_dataset=tc.cat(new_dataset,dim=0)
    new_label=np.array(new_label)
    return new_dataset,new_label

def label_to_img(imgwidth, imgheight, w, h, labels):
    """
    Transform list of labels to image.
    """
    im = np.zeros([imgwidth, imgheight])
    m = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

def make_img_overlay(img, predicted_img):
    """
    Combines image and prediction.
    """
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

def compute_F1(Y,Z):
    """
    Compute F1 metric when inputs are LISTs
    Input:  Y, true value assigned to the patch.
            Z, list of prediction for patches
    """
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    matrix = []
    for i in range(len(Y)):
        if (round(Y[i])==0) & (Z[i]==0):
            TN = TN + 1
        elif (round(Y[i])==1) & (Z[i]==0):
            FN = FN + 1   
        elif (round(Y[i])==1) & (Z[i]==1):
            TP = TP + 1  
        else:
            FP = FP + 1

    F1_score = 0
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2*precision*recall / (precision+recall)
    except:
        print('Something goes wrong...')
    
    return F1_score
    
def calcul_F1(mask, prediction):   
    """
    Compute F1 metric when inputs are MATRICES
    Input: mask, true values 
           prediction, predicted values.
    """
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (round(mask[i,j])==0) & (prediction[i,j]==0):
                TN = TN + 1
            elif (round(mask[i,j])==1) & (prediction[i,j]==0):
                FN = FN + 1   
            elif (round(mask[i,j])==1) & (prediction[i,j]==1):
                TP = TP + 1  
            else:
                FP = FP + 1
    
    F1_score = 0
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1_score = 2*precision*recall / (precision+recall)
    except:
        print('Something goes wrong...')
        
    return F1_score