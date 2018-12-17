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
from dataset import *

def print_image_test(test_path, img_nb, model, do_prep=True):
    ''' take a 608 * 608 image'''
    dataset = Testset(test_path, 50, do_prep = do_prep, normalize=True)
    model.eval()
    test_image,original_image = dataset.__getitem__(img_nb)
    test_image=test_image.view(-1,dataset.get_features(),608,608)
    predictions=model(test_image).detach().numpy().reshape(608,608)
    predictions = 1*(predictions > 0.5)
    #predictions = post_processing(predictions,27,8,3,3,38)
    image = make_img_overlay(original_image, predictions)
    
    plt.figure(figsize=(10,10))
    plt.imshow(image)

def test_on_image(test_path, img_nb, model, do_prep=True):
    ''' take a 608 * 608 image'''
    dataset = Testset(test_path, 50, do_prep = do_prep, normalize=True)
    test_image,_ = dataset.__getitem__(img_nb)
    if tc.cuda.is_available():
        model.cuda()
        test_image = test_image.cuda()
    model.eval()
    test_image=test_image.view(-1,dataset.get_features(),608,608)
    predictions=model(test_image).cpu().detach().numpy().reshape(608,608)
    predictions = 1*(predictions > 0.5)
    return predictions


def create_submission_UNet(test_path, model, name_submission, do_preprocessing= False):
    ''' create the submission file for U-Net.
    
    test_set is a list of images'''
    
    if tc.cuda.is_available():
        model.cuda()
    model=model.eval()
    tc.no_grad()
    list_of_mask = []
    for i in tqdm(range(50)):
        list_of_mask.append(test_on_image(test_path,i, model,do_prep=True))
                
        
    #mask = (mask.detach().numpy() > 0.5)*1
    list_of_names= []
    for i in range(50):
        plt.imsave( "prediction_"+str(i+1)+".png", list_of_mask[i], cmap = matplotlib.cm.gray)
        list_of_names.append("prediction_"+str(i+1)+".png")
    masks_to_submission(name_submission, *list_of_names)