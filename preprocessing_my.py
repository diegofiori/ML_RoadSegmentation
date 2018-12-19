import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import cv2
from scipy import ndimage
from PIL import Image
from helpers_img_my import *
from sklearn import preprocessing as prp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans


def rotation(orig, gts, diagonal=False):
    ks=[45,90,135]
    rotated=[ndimage.rotate(img,k) for k in ks for img in orig]
    gt_rotated=[ndimage.rotate(gt_img,k) for k in ks for gt_img in gts]
    orig=orig+rotated
    gts=gts+gt_rotated
    if diagonal:
        rotated=[ndimage.rotate(img,45,reshape=False,mode='reflect') for img in orig]
        gt_rotated = [ndimage.rotate(gt_img,45,reshape=False,mode='reflect') for gt_img in gts]
        orig = orig + rotated
        gts = gts + gt_rotated  
    return orig,gts

def flip(orig,gts):
    rotated=[cv2.flip(img,1) for img in orig]
    gt_rotated=[cv2.flip(gt_img,1) for gt_img in gts]
    orig=orig+rotated
    gts=gts+gt_rotated
    return orig,gts

def expand(orig,gts):
    ks=[45,90,135]
    for k in ks:
    rotated=[ndimage.rotate(img,k) for k in ks for img in orig]
    gt_rotated=[ndimage.rotate(gt_img,k) for k in ks for gt_img in gts]
    flipped=[cv2.flip(img,1) for img in orig]
    gt_flipped=[cv2.flip(gt_img,1) for gt_img in gts]
    orig=orig+rotated
    orig=orig+flipped
    gts=gts+gt_rotated
    gts=gts+gt_flipped
    return orig,gts

def add_gray_dimension(img):
    out=np.dot(img[...,:3], [0.299, 0.587, 0.114])
    shape_one=[out.shape[0], out.shape[1], 1]
    out = np.reshape(out, shape_one)
    return out

def add_laplacian(img):
    laplbew=ndimage.gaussian_laplace(add_gray_dimension(img),2)
    lapl=ndimage.gaussian_laplace(img,2)
    return laplbew,lapl

def add_sobel(img):
    sx = ndimage.sobel(img, axis=0, mode='constant')
    sy = ndimage.sobel(img, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    return sob

def add_segment(im):
    n = 10
    l = 256
    im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    mask = (im > im.mean()).astype(np.float)
    mask += 0.1 * im
    img = mask + 0.2*np.random.randn(*mask.shape)
    hist, bin_edges = np.histogram(img, bins=60)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    binary_img = img > 0.5
    open_img = ndimage.binary_opening(binary_img)
    # Remove small black hole
    close_img = ndimage.binary_closing(open_img)
    close_img=add_gray_dimension(close_img)
    return close_img

def add_label_kmeans(img,n_cluster, max_iters, threshold):
    """Using k-means to generate a new feature.
       INPUT: path of the image"""
    original_image = img
    x,y,z = original_image.shape
    processed_image = original_image.reshape(x*y,z)
    model = KMeans(n_clusters=n_cluster, random_state=2, init = 'k-means++', n_init = 5).fit(processed_image)
    assignments = model.labels_
    mu = model.cluster_centers_
    new_image = processed_image.reshape(x,y,z)  
    assignments = assignments.reshape(x,y)
    final_img = np.concatenate((new_image,assignments[:,:,np.newaxis]),axis=2)
    return final_img

def add_features(img):
    gray_img = add_gray_dimension(img)
    sob = add_sobel(img)
    lapbew,lap=add_laplacian(img)
    seg=add_segment(img)
    img = np.concatenate((img, gray_img), axis = 2)
    img = np.concatenate((img, sob), axis = 2)
    img = np.concatenate((img, lapbew), axis = 2)
    img = np.concatenate((img, lap), axis = 2)    
    img = np.concatenate((img, seg), axis = 2)
    return img

def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def poly_features(feats,deg):
    """ Performs feature augmentations by taking the polynomials and the interactions of the features."""
    poly = PolynomialFeatures(deg)
    feats = poly.fit_transform(feats.reshape(1,-1))
    feats = feats.reshape(-1,)
    return feats


