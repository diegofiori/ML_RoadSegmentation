import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from scipy import ndimage
from PIL import Image
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing as prp
from helpers_img_my import *
from Post_processing import *
#from helpers_img import *
from sklearn import model_selection as modsel
from preprocessing_my import *


def obtain_splits(k,ims):
    """
    Returns the indices for train and test sets in a K-fold cross-validation.
    Input: k, parameter K of the cross-validation;
           ims, set of images. 
    """
    kf=modsel.KFold(n_splits=k_fold,shuffle=True)
    kf.get_n_splits(ims)
    splits=kf.split(ims)
    trains=[]
    tests=[]
    for (train_id,test_id) in splits:
        trains.append(train_id)
        tests.append(test_id)
        
    return trains,tests


def preprocessed(ims,gt_ims,patch_size,deg):
    """
    Returns the preprocessed images and ground-truth images on which to perform cross-validation.
    Input: ims, set of images;
           gt_ims: set of ground-truth images;
           patch_size: size of a patch;
           deg: degree of the polynomial taken.
    Output: features corresponding to patches after expanding the dataset with rotations and flips.
                Fetaures are expanded and raised to powers.
    """
    ims=[add_features(ims[i]) for i in range(len(ims))]
    ims,gt_ims = rotation(ims,gt_ims)
    ims,gt_ims = flip(ims,gt_ims)
    ims = [img_crop(ims[i], patch_size, patch_size) for i in range(len(ims))]
    gt_ims = [img_crop(gt_ims[i], patch_size, patch_size) for i in range(len(gt_ims))]

    ims = np.asarray([ims[i][j] 
                        for i in range(len(ims)) 
                        for j in range(len(ims[i]))])

    gt_ims = np.asarray([gt_ims[i][j] 
                        for i in range(len(gt_ims)) 
                        for j in range(len(gt_ims[i]))])

    ims=[extract_features(ims[i]) for i in range(len(ims))]
    ims = np.asarray([poly_features(ims[i],deg) for i in range(len(ims))])
    gt_ims = np.asarray([value_to_class(np.mean(gt_ims[i])) for i in range(len(gt_ims))])
    
    
    