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
from helpers_img import *


# Settings
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16 # each patch is 16*16 pixels

#
# TRAINING SET
#

root_dir = "training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
n = min(75, len(files))
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_dir = root_dir + "groundtruth/"
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]

# Rotate the images
imgs,gt_imgs = rotation(imgs,gt_imgs)
# Flip the imgs
imgs,gt_imgs = flip(imgs,gt_imgs)
# Add features
imgs_augm=[add_features(imgs[i]) for i in range(len(imgs))]

# Work on patches
img_patches = [img_crop(imgs_augm[i], patch_size, patch_size) for i in range(len(imgs_augm))]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(len(gt_imgs))]
img_patches = np.asarray([img_patches[i][j] 
                          for i in range(len(img_patches)) 
                          for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] 
                          for i in range(len(gt_patches)) 
                          for j in range(len(gt_patches[i]))])

# Obtain regressors
deg=3
img_patches=[extract_features(img_patches[i]) for i in range(len(img_patches))]
X = np.asarray([poly_features(img_patches[i],deg) for i in range(len(img_patches))])
Y = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

# Fit the model and predict
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
logreg.fit(X, Y)
Z=logreg.predict(X)
print('F1_score on the training set= ' + str(compute_F1(Y, Z)))


#
# TEST SET
#

root = "training/"
image_dir = root + "images/"
files = os.listdir(image_dir)
imgs_te = [load_image(image_dir + files[i]) for i in np.arange(n+1,len(files))]
gt_dir = root_dir + "groundtruth/"
gt_imgs_te = [load_image(gt_dir + files[i]) for i in np.arange(n+1,len(files))]

# Add features
imgs_te_aug=[add_features(imgs_te[i]) for i in range(len(imgs_te))]

# Work on patches
img_patches_te = [img_crop(imgs_te_aug[i], patch_size, patch_size) for i in range(len(imgs_te_aug))]
gt_patches_te = [img_crop(gt_imgs_te[i], patch_size, patch_size) for i in range(len(gt_imgs_te))]
img_patches_te = np.asarray([img_patches_te[i][j] 
                             for i in range(len(img_patches_te)) 
                             for j in range(len(img_patches_te[i]))])
gt_patches_te =  np.asarray([gt_patches_te[i][j] 
                             for i in range(len(gt_patches_te)) 
                             for j in range(len(gt_patches_te[i]))])


# Obtain regressors
deg=3
img_patches_te=[extract_features(img_patches_te[i]) for i in range(len(img_patches_te))]
X_te = np.asarray([poly_features(img_patches_te[i],deg) for i in range(len(img_patches_te))])
Y_te = np.asarray([value_to_class(np.mean(gt_patches_te[i])) for i in range(len(gt_patches_te))])

# Predict
Z_te = logreg.predict(X_te)
print('F1_score on the test set before post-processing= ' + str(compute_F1(Y_te, Z_te)))

# Post-processing
Z_pp=[]
for i in range(len(gt_patches_te)):
    Z_pp = Z_pp + post_processing(Z_te[i*625:(i+1)*625],18,9,3,3)

print('F1_score after post-processing= ' + str(compute_F1(Y_te, Z_pp)))


