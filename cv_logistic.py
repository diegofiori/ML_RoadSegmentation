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
from preprocessing_my import *

# Settings
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16 # each patch is 16*16 pixels
deg=2

root_dir = "training/"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
files = os.listdir(image_dir)

n=len(files)
#n=4
k_fold=3 #number of folds in the k-fold cross-validation

# Retrieve the set of images and gt_images
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

seed = 1
np.random.seed(seed)

# K-Fold Cross Validation
kf=modsel.KFold(n_splits=k_fold,shuffle=True)
kf.get_n_splits(imgs)

# Define list of lambda to regularize
lambdas=np.logspace(0,8,15) # 1/lambda

# Create the vector to store the error for each value of lambda
mean_f1=np.zeros(lambdas.size)

# For each value of lambda
for id_lam,lam in enumerate(lambdas):
    
    # Store for a fixed lambda the error
    nb_f1_te = np.zeros(k_fold)
    for ind,[train_id,test_id] in enumerate(kf.split(imgs)):
        
        # Split the dataset
        im_tr=[imgs[idx] for idx in train_id]
        im_te=[imgs[idx] for idx in test_id]
        gt_tr=[gt_imgs[idx] for idx in train_id]
        gt_te=[gt_imgs[idx] for idx in test_id]
        
        im_tr,gt_tr = rotation(im_tr,gt_tr)
        im_tr,gt_tr = flip(im_tr,gt_tr)
        im_tr=[add_features(im_tr[i]) for i in range(len(im_tr))]
        im_te=[add_features(im_te[i]) for i in range(len(im_te))]

        
        img_patches_tr = [img_crop(im_tr[i], patch_size, patch_size) for i in range(len(im_tr))]
        img_patches_te = [img_crop(im_te[i], patch_size, patch_size) for i in range(len(im_te))]
        gt_patches_tr = [img_crop(gt_tr[i], patch_size, patch_size) for i in range(len(gt_tr))]
        gt_patches_te = [img_crop(gt_te[i], patch_size, patch_size) for i in range(len(gt_te))]
        
        im_patches_tr = np.asarray([img_patches_tr[i][j] 
                          for i in range(len(img_patches_tr)) 
                          for j in range(len(img_patches_tr[i]))])
        im_patches_te = np.asarray([img_patches_te[i][j] 
                          for i in range(len(img_patches_te)) 
                          for j in range(len(img_patches_te[i]))])
        gt_patches_tr =  np.asarray([gt_patches_tr[i][j] 
                          for i in range(len(gt_patches_tr)) 
                          for j in range(len(gt_patches_tr[i]))])
        gt_patches_te =  np.asarray([gt_patches_te[i][j] 
                          for i in range(len(gt_patches_te)) 
                          for j in range(len(gt_patches_te[i]))])
        
        im_patches_tr=[extract_features(im_patches_tr[i]) for i in range(len(im_patches_tr))]
        im_patches_te=[extract_features(im_patches_te[i]) for i in range(len(im_patches_te))]

        # Create the variables X and Y associated to the split
        X_tr = np.asarray([poly_features(im_patches_tr[i],deg) for i in range(len(im_patches_tr))])
        Y_tr = np.asarray([value_to_class(np.mean(gt_patches_tr[i])) for i in range(len(gt_patches_tr))])
        
        X_te = np.asarray([poly_features(im_patches_te[i],deg) for i in range(len(im_patches_te))])
        Y_te = np.asarray([value_to_class(np.mean(gt_patches_te[i])) for i in range(len(gt_patches_te))])
        
        # Make the prediction
        logreg = linear_model.LogisticRegression(C=lam, class_weight="balanced")
        logreg.fit(X_tr, Y_tr)

        Z_te = logreg.predict(X_te)

        # Apply the post processing
        Z_pp=[]
        for i in range(len(gt_patches_te)):
            Z_pp = Z_pp + post_processing(Z_te[i*625:(i+1)*625],18,9,3,3,25)
            
        # Calculate F1 and store it
        nb_f1_te[ind]=compute_F1(Y_te, Z_pp)
        
    mean_f1[id_lam]=nb_f1_te.mean()
    
# Find the best lambda
best_lambda=lambdas[np.argmax(mean_f1)]


f1_cv=max(mean(f1))


