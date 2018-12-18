import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import cv2
from scipy import ndimage
from PIL import Image
from helpers_img import *

def rotation(orig, gts, diagonal=False):
    ks=[90,180,270]
    rotated=[ndimage.rotate(img,k) for img in orig for k in ks]
    gt_rotated=[ndimage.rotate(gt_img,k) for gt_img in gts for k in ks]
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
    img = add_label_kmeans(img,25, 100, 1e-6)
    return img

def extract_features(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

def poly_features(feats,deg):
    """
    Performs feature augmentations by taking the polynomials and the interactions of the features.
    """
    poly = prp.PolynomialFeatures(deg)
    feats = poly.fit_transform(feats.reshape(1,-1))
    feats = feats.reshape(-1,)
    return feats

def add_border(imgs,new_size):
    old_size = imgs.shape[0]
    add = int((new_size-old_size)/2)
    if add>0:
        new_im = imgs[:add,:,:]
        new_im = new_im[::-1,:,:]
        final_row = np.concatenate((new_im,imgs[:,:]),axis=0)
        new_im = imgs[-add:,:,:]
        new_im = new_im[::-1,:,:]
        final_row = np.concatenate((final_row,new_im[:,:]),axis=0)
        new_im = final_row[:,:add,:]
        new_im = new_im[:,::-1,:]
        final = np.concatenate((new_im,final_row[:,:]),axis=1)
        new_im = final_row[:,-add:,:]
        new_im = new_im[:,::-1,:]
        final = np.concatenate((final,new_im[:,:]),axis=1)
    else:
        final = imgs
    return final

def add_label_kmeans(img,n_cluster, max_iters, threshold):
    """Using k-means to generate a new feature.
       INPUT: path of the image"""
    original_image = img
    x,y,z = original_image.shape
    processed_image = original_image.reshape(x*y,z)
    model = KMeans(n_clusters=n_cluster, random_state=2, init = 'k-means++', n_init = 2).fit(processed_image)
    assignments = model.labels_
    mu = model.cluster_centers_
    new_image = processed_image.reshape(x,y,z)  
    assignments = assignments.reshape(x,y)
    final_img = np.concatenate((new_image,assignments[:,:,np.newaxis]),axis=2)
    return final_img