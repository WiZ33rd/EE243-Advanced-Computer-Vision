import os
import scipy.io as io
import sys
import torchreid
import numpy as np
import torch
from torchvision import transforms


if __name__ == '__main__':   

    ###########################################################################    
    ################### DO NOT CHANGE ANYTHING IN THIS BLOCK ##################
    ###########################################################################

    v_idx = int(sys.argv[1])

    I = io.loadmat('./video.mat')['I']
    img = I(:, :, :, v_idx)
    try:
        bbox = io.loadmat('./bbox.mat')['bbox1']
    except KeyError:
        bbox = io.loadmat('./bbox.mat')['bbox2']

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    # print(device)

    model = torchreid.models.build_model(name='resnet50',
                                     num_classes=10,
                                     loss='softmax',
                                     pretrained=True
                                    )
    
    ###########################################################################



    ###########################################################################
    ########################### FILL YOUR CODE WILL ###########################
    ###########################################################################

    # DEFINE TRANSFORM TO CONVERT TO TENSOR
    transform = 

    # FILL CODE FOR FEATURE EXTRACTION FROM PENULTIMATE LAYER.
    # YOU NEED TO REMOVE LAST LAYER OF MODEL DEFINED ABOVE. 
    # AFTER REMOVING LAST LAYER YOUR MODEL WILL OUTPUT A FEATURE
    # OF LENGTH 2048
    extractor = 


    # DEFINE FEATURES ARRAY BASED ON SIZE OF BOUNDING BOX
    feat = 


    # FOR EACH BOUNDING BOX EXTRACT FEATURES   
    for idx in range(feat.shape[0]):
        #LOAD ONE BOUNDING BOX AT A TIME
        
        # CROP THE INPUT FRAME BASED ON BOUNDING BOX
        img_crop = 

        # EXTRACT FEATURE FOR THE CROPPED IMAGE AND SAVE IN THE 
        feat[idx, :] = 
   
    ###########################################################################
    
    io.savemat('./feat.mat', {'feat': feat})
