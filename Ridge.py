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
from helpers_regression import *

# Settings and Reading
foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch
patch_size = 16 # each patch is 16*16 pixels
deg=2
root_dir = "training/"
image_dir = root_dir + "images/"
gt_dir = root_dir + "groundtruth/"
files = os.listdir(image_dir)
n=len(files)
#n=4
k_fold=2 #number of folds in the k-fold cross-validation

ims = [load_image(image_dir + files[i]) for i in range(n)]
gt_ims = [load_image(gt_dir + files[i]) for i in range(n)]

seed = 1
np.random.seed(seed)

trains,tests=obtain_splits(k_fold,ims)