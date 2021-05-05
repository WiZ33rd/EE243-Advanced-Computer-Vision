import os
import scipy.io as io
import sys
import torchreid
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


if __name__ == '__main__':

    ###########################################################################
    ################### DO NOT CHANGE ANYTHING IN THIS BLOCK ##################
    ###########################################################################

    v_idx = int(sys.argv[1])

    I = io.loadmat('./video.mat')['I']
    img = I[:, :, :, v_idx]
    try:
        bbox = io.loadmat('./bbox.mat')['bbox1']
    except KeyError:
        bbox = io.loadmat('./bbox.mat')['bbox2']

    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    ])
    # FILL CODE FOR FEATURE EXTRACTION FROM PENULTIMATE LAYER.
    # YOU NEED TO REMOVE LAST LAYER OF MODEL DEFINED ABOVE.
    # AFTER REMOVING LAST LAYER YOUR MODEL WILL OUTPUT A FEATURE
    # OF LENGTH 2048
    extractor = torch.nn.Sequential(*list(model.children())[: -1])
    extractor.to(device)

    # DEFINE FEATURES ARRAY BASED ON SIZE OF BOUNDING BOX
    feat = np.zeros((bbox.shape[0], 2048))
    round_bbox = np.around(bbox)
    round_bbox = round_bbox.astype(int)

    # FOR EACH BOUNDING BOX EXTRACT FEATURES
    for idx in range(feat.shape[0]):
        # LOAD ONE BOUNDING BOX AT A TIME

        # CROP THE INPUT FRAME BASED ON BOUNDING BOX
        hrange = min(round_bbox[idx, 1]+round_bbox[idx, 3], I.shape[0])
        vrange = min(round_bbox[idx, 0]+round_bbox[idx, 2], I.shape[1])
        img_crop = img[round_bbox[idx, 1]:hrange, round_bbox[idx, 2]:vrange, :]

        # EXTRACT FEATURE FOR THE CROPPED IMAGE AND SAVE IN THE
        PIL_image = Image.fromarray(img_crop)
        x = transform(PIL_image)
        x = torch.unsqueeze(x, 0)
        x = x.to(device)
        output = extractor(x)
        feature_array = output.cpu()
        feature_array = feature_array.detach().numpy()
        feat[idx, :] = feature_array[0, :, 0, 0]

    ###########################################################################

    io.savemat('./feat.mat', {'feat': feat})
