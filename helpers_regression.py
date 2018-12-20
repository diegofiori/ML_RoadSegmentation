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
from helpers_img import *
from Post_processing import *
from sklearn import model_selection as modsel
from preprocessing import *


def post_processing(label,threshold,size_min,verbarg,horbarg,size_image):
    '''Choose the functions to post_process our predictions'''
    label = complete_lines(label,threshold)
    label = remove_isolated_connected_component(label,size_min)
    label = clean_garbage_vert(label,verbarg,size_image)
    label = clean_garbage_hor(label,horbarg,size_image)
    label = remove_isolated_connected_component(label,size_min)
    return label

def obtain_splits(k_fold,ims):
    """
    Returns the indices for train and test sets in a K-fold cross-validation.
    Input: k, parameter K of the cross-validation;
           ims, set of images. 
    """
    # Create the folder to operate cross validation
    kf=modsel.KFold(n_splits=k_fold,shuffle=True)
    kf.get_n_splits(ims)
    splits=kf.split(ims)
    # Create the test and train set for cross validation
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
    # Preprocess images
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
    return ims,gt_ims
    
def cross_validation_logistic(lambdas,k_fold,trains,tests,ims,gt_ims):
    '''Return best lambda by cross validation
       Input: lambdas, list of lambda;
              k_fold: number of k_fold;
              trains: set of trains id
              tests: set of tests id
              ims, set of images;
              gt_ims: set of ground-truth images;
       Output: best lambda.'''
    n = len(ims)
    for id_lam,lam in enumerate(lambdas):
        print('lambda: '+str(id_lam))
        nb_f1_te = np.zeros(k_fold)
        int_=0
        for train,test in zip(trains,tests):
            # Create a matrix for each patch of each image the list of features
            im_tr = np.zeros((len(train)*625*8,325))
            gt_tr = np.zeros((len(train)*625*8))
            inter = 0
            for idx in train:
                # Return a length 8 list of arrays 
                temps = [ims[625*idx + (625*n)*k: 625*(idx+1) + (625*n)*k] for k in np.arange(0,8)]
                # Return a length 8 list of arrays
                gt_temps = [gt_ims[625*idx + (625*n)*k :625*(idx+1) + (625*n)*k] for k in np.arange(0,8)]
                # Create the two matrices
                for (temp,gt_temp) in zip(temps,gt_temps):
                    im_tr[625*inter:625*(inter+1),:]= temp
                    gt_tr[625*inter:625*(inter+1)]=gt_temp
                    inter = inter + 1
            
            # Same for test
            im_te = np.zeros((len(train)*625,325))
            gt_te = np.zeros((len(train)*625))
            inter = 0
            for idx in test:
                im_te[625*inter:625*(inter+1),:] = ims[625*idx : 625*(idx+1)]
                gt_te[625*inter:625*(inter+1)] = gt_ims[625*idx : 625*(idx+1)]
                inter = inter + 1
        
            # Operate logistic regression
            logreg = linear_model.LogisticRegression(C=lam, class_weight="balanced")
            logreg.fit(im_tr, gt_tr)

            Z_te = logreg.predict(im_te)
        
            # Post process the image
            Z_pp=[]
            for i in range(len(gt_te)):
                Z_pp = Z_pp + post_processing(Z_te[i*625:(i+1)*625],18,9,3,3,25)
            
            nb_f1_te[int_]=compute_F1(gt_te, Z_pp)
            print(nb_f1_te[int_])
            int_=int_+1
    
        # Calculate the mean error for each lambda
        mean_f1[id_lam]=nb_f1_te.mean()
    
    # Return the best lambda
    best_lambda=lambdas[np.argmax(mean_f1)]
    return best_lambda
    
def cross_validation_ridge(lambdas,thresh,k_fold,trains,tests,ims,gt_ims):
    '''Return best lambda and threshold by cross validation
       Input: lambdas, list of lambda;
              k_fold: number of k_fold;
              trains: set of trains id
              tests: set of tests id
              ims, set of images;
              gt_ims: set of ground-truth images;
       Output: best lambda.'''
    n = len(ims)
    for id_lam,lam in enumerate(lambdas):
        print('lambda: '+str(id_lam))
        for id_t,t in enumerate(thresh):
            print('thresh: '+str(id_t))
            nb_f1_te = np.zeros(k_fold)
            ind=0
            for train,test in zip(trains,tests):
                # Create a matrix for each patch of each image the list of features
                im_tr = np.zeros((len(train)*625*8,325))
                gt_tr = np.zeros((len(train)*625*8))
                inter = 0
                for idx in train:
                    # Return a length 8 list of arrays 
                    temps = [ims[625*idx + (625*n)*k: 625*(idx+1) + (625*n)*k] for k in np.arange(0,8)]
                    # Return a length 8 list of arrays
                    gt_temps = [gt_ims[625*idx + (625*n)*k :625*(idx+1) + (625*n)*k] for k in np.arange(0,8)]
                    for (temp,gt_temp) in zip(temps,gt_temps):
                        im_tr[625*inter:625*(inter+1),:]= temp
                        gt_tr[625*inter:625*(inter+1)]=gt_temp
                        inter = inter + 1
            
                im_te = np.zeros((len(train)*625,325))
                gt_te = np.zeros((len(train)*625))
                inter = 0
                for idx in test:
                    im_te[625*inter:625*(inter+1),:] = ims[625*idx : 625*(idx+1)]
                    gt_te[625*inter:625*(inter+1)] = gt_ims[625*idx : 625*(idx+1)]
                    inter = inter + 1
           
                ridgereg = linear_model.Ridge(alpha=lam, normalize=True)
                ridgereg.fit(im_tr, gt_tr)

                Z_te = ridgereg.predict(im_te)
                Z_te=1*(Z_te>t)
                Z_pp=[]
                for i in range(len(gt_te)):
                    Z_pp = Z_pp + post_processing(Z_te[i*625:(i+1)*625],18,9,3,3,25)

                nb_f1_te[ind]=compute_F1(gt_te, Z_pp)
                print(nb_f1_te[ind])
                ind=ind+1
        
            mean_f1[id_lam,id_t]=nb_f1_te.mean()
    
    best=np.max(mean_f1)
    ids=np.where(mean_f1==best)
    best_lambda=lambdas[ids[0][0]]
    best_thresh=thresh[ids[1][0]]
    
    return best_lambda,best_thresh