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

def plot_image(root_dir, nb_image, model):
    ''' plot the desired image from the test set.
    root_dir: test images directory
    nb_image: nb of the desired image
    model: model already trained'''
    dataset = TestsetDeepNet(root_dir,50)
    image, original = dataset.__getitem__(nb_image)
    if tc.cuda.is_available():
        model.cuda()
        image=image.cuda()
    model.eval()
    prediction = model(image).detach().cpu().numpy().reshape(-1,)
    prediction = 1*(prediction>0.5)
    model.cpu()
    print(prediction.shape)
    prediction = label_to_img(608, 608, 16, 16, prediction)
    #prediction=post_processing(prediction,32,9,3,3)
    
    original  = make_img_overlay(original, prediction)
    
    plt.figure(figsize=(10,10))
    plt.imshow(original)

    

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

def create_submission_DeepNet(root_dir, model, name_file, do_postprocess=False):
    ''' the function takes as input the test data and the models used for prediction. 
    If a list of model is given, the prediction will be done with majority vote. 
    
    The function is written explicitly for prediction using SimpleNet model.
    
    test_data: list of images.
    
    models: list of models or single model
    
    w, h: width and high of the patches'''
    
    # from list to Tensor
    dataset = TestsetDeepNet(root_dir,50)
    test_loader = DataLoader(dataset)
    predictions = []
    if tc.cuda.is_available():
        model.cuda()
    for test_data,_ in tqdm(test_loader):
        test_data = test_data.view(-1,3,48,48)
        if tc.cuda.is_available():
            test_data=test_data.cuda()
        prediction = model(test_data).detach().cpu().numpy()
        
        prediction = 1*(prediction > 0.5)
     
        predictions.append(prediction.reshape(-1,))
    if do_postprocess:
        for i in range(len(predictions)):
            predictions[i]=complete_lines(predictions[i],35)
            predictions[i] = remove_isolated_connected_component(predictions[i],3)

    # from patch to image
    list_of_masks=[label_to_img(608, 608, 16, 16, predictions[k]) for k in range(50)]
    list_of_string_names = []
    for i in range(50):
        plt.imsave( "prediction_"+str(i+1)+".png", list_of_masks[i], cmap = matplotlib.cm.gray)
        list_of_string_names.append("prediction_" + str(i+1) + ".png")
    # create file submission
    masks_to_submission(name_file, *list_of_string_names)