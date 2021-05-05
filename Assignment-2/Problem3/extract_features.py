import os
import numpy as np
import scipy.io as sio
import glob
import torchvision
from models.alexnet import alexnet
from models.vgg import vgg16
import torchvision.transforms as transforms
from PIL import Image
import torch


def main():
    alex_feature = []
    alex_label = []

    vgg16_feature = []
    vgg16_label = []

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True,
                                              transform=transform)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True,
                                             transform=transform)

    # [Problem 4 a.] IMPORT VGG16 AND ALEXNET FROM THE MODELS FOLDER WITH
    # PRETRAINED = TRUE
    vgg16_extractor = vgg16(pretrained=True)
    vgg16_extractor.eval()

    alex_extractor = alexnet(pretrained=True)
    alex_extractor.eval()

    for idx, data in enumerate(train_data):

        image, label = data

        # [Problem 4 a.] OUTPUT VARIABLE F_vgg and F_alex EXPECTED TO BE THE
        # FEATURE OF THE IMAGE OF DIMENSION (4096,) AND (256,), RESPECTIVELY.
        image = torch.unsqueeze(image, 0)

        F_vgg = vgg16_extractor(image)
        vgg16_feature.append(F_vgg.detach().numpy()[0])
        vgg16_label.append(label)

        F_alex = alex_extractor(image)
        alex_feature.append(F_alex.detach().numpy()[0])
        alex_label.append(label)

    sio.savemat('vgg16_train.mat', mdict={'feature': vgg16_feature,
                                          'label': vgg16_label})
    sio.savemat('alexnet_train.mat', mdict={'feature': alex_feature,
                                            'label': alex_label})

    # 1. EXTRACT FEATURES USING THE MODELS - ALEXNET AND VGG16
    test_vgg16_feature = []
    test_vgg16_label = []
    test_alex_feature = []
    test_alex_label = []
    for idx, data in enumerate(test_data):
        test_image, test_label = data
        test_image = torch.unsqueeze(test_image, 0)

        # 1. # EXTRACT FEATURES USING THE MODELS - ALEXNET AND VGG16
        F_test_vgg16 = vgg16_extractor(test_image).detach().numpy()[0]
        test_vgg16_feature.append(F_test_vgg16)
        test_vgg16_label.append(test_label)

        F_test_alex = alex_extractor(test_image).detach().numpy()[0]
        test_alex_feature.append(F_test_alex)
        test_alex_label.append(test_label)

    sio.savemat('vgg16_test.mat', mdict={'feature': test_vgg16_feature,
                                         'label': test_vgg16_label})
    sio.savemat('alexnet_test.mat', mdict={'feature': test_alex_feature,
                                           'label': test_alex_label})


# Normalize feature by colum uniformly
def feature_normalize(feature):
    for i in range(feature.shape[1]):
        if (max(feature[:, i]) - min(feature[:, i])) != 0.0:
            feature[:, i] = ((feature[:, i] - min(feature[:, i])) /
                             (max(feature[:, i]) - min(feature[:, i])))
    return feature


# predict by model from testset
def pred(vgg_distance, alex_distance, test_vgg_label, test_alex_label,
         train_vgg_label, train_alex_label, K, i):

    min_vgg_distance_idx = np.argsort(vgg_distance)[:K]
    min_alex_distance_idx = np.argsort(alex_distance)[:K]

    vgg_neighbors_label = train_vgg_label[min_vgg_distance_idx]
    alex_neighbors_label = train_alex_label[min_alex_distance_idx]

    # Predict by find most frequent labels
    vgg_count = np.bincount(vgg_neighbors_label)
    alex_count = np.bincount(alex_neighbors_label)

    # To avoid all different K labels
    if(max(vgg_count) == 1):
        vgg_pred = train_vgg_label[min_vgg_distance_idx[0]]
    else:
        vgg_pred = np.argmax(vgg_count)
    if(max(alex_count) == 1):
        alex_pred = train_alex_label[min_alex_distance_idx[0]]
    else:
        alex_pred = np.argmax(alex_count)
    return (vgg_pred == test_vgg_label[i]), (alex_pred == test_alex_label[i])


def KNN_test():
    # FILL IN TO LOAD THE SAVED .MAT FILE

    # Load feature and labels from .mat file
    train_vgg_mat = sio.loadmat('vgg16_train.mat')
    train_vgg_feature = np.asarray(train_vgg_mat['feature'])
    train_vgg_label = train_vgg_mat['label'][0]

    train_alex_mat = sio.loadmat('alexnet_train.mat')
    train_alex_feature = np.asarray(train_alex_mat['feature'])
    train_alex_label = train_alex_mat['label'][0]

    test_vgg_mat = sio.loadmat('vgg16_test.mat')
    test_vgg_feature = np.asarray(test_vgg_mat['feature'])[:10, ]
    test_vgg_label = test_vgg_mat['label'][0][:10]

    test_alex_mat = sio.loadmat('alexnet_test.mat')
    test_alex_feature = np.asarray(test_alex_mat['feature'])[:10, ]
    test_alex_label = test_alex_mat['label'][0][:10]

    train_size = train_vgg_feature.shape[0]
    test_size = test_vgg_feature.shape[0]

    # Normalize feature to avoid scale problem
    train_vgg_feature = feature_normalize(train_vgg_feature)
    train_alex_feature = feature_normalize(train_alex_feature)
    test_vgg_feature = feature_normalize(test_vgg_feature)
    test_alex_feature = feature_normalize(test_alex_feature)

    # 2. FIND NEAREST NEIGHBOUT OF THIS FEATURE FROM FEATURES STORED IN ALEXNET.MAT AND VGG16.MAT
    vgg16_correct_1 = 0
    alex_correct_1 = 0
    vgg16_correct_3 = 0
    alex_correct_3 = 0
    vgg16_correct_5 = 0
    alex_correct_5 = 0

    for i in range(test_size):
        vgg_distance = []
        alex_distance = []
        # calculate the euclidean distance of train feature and test feature
        for j in range(train_size):
            vgg_distance.append(np.linalg.norm(train_vgg_feature[j, ] -
                                               test_vgg_feature[i, ]))
            alex_distance.append(np.linalg.norm(train_alex_feature[j, ] -
                                                test_alex_feature[i, ]))

        # find K nearest neighbors and their labels
        vgg16_1, alex_1 = pred(vgg_distance, alex_distance, test_vgg_label,
                               test_alex_label, train_vgg_label,
                               train_alex_label, 1, i)
        vgg16_3, alex_3 = pred(vgg_distance, alex_distance, test_vgg_label,
                               test_alex_label, train_vgg_label,
                               train_alex_label, 3, i)
        vgg16_5, alex_5 = pred(vgg_distance, alex_distance, test_vgg_label,
                               test_alex_label, train_vgg_label,
                               train_alex_label, 5, i)
        vgg16_correct_1 += vgg16_1
        vgg16_correct_3 += vgg16_3
        vgg16_correct_5 += vgg16_5
        alex_correct_1 += alex_1
        alex_correct_3 += alex_3
        alex_correct_5 += alex_5
        if (i % 100) == 99:
            print('current index', i, 'correct', vgg16_correct_1,
                  alex_correct_1, vgg16_correct_3, alex_correct_3,
                  vgg16_correct_5, alex_correct_5)

    # 3. COMPUTE ACCURACY
    vgg16_accuracy_1 = vgg16_correct_1*100/test_size
    alex_accuracy_1 = alex_correct_1*100/test_size
    vgg16_accuracy_3 = vgg16_correct_3*100/test_size
    alex_accuracy_3 = alex_correct_3*100/test_size
    vgg16_accuracy_5 = vgg16_correct_5*100/test_size
    alex_accuracy_5 = alex_correct_5*100/test_size
    print('When K=1, the accuracy of VGG16 is %.2f %%, the accuracy of AlexNet is %.2f %%.' % (
        vgg16_accuracy_1, alex_accuracy_1))
    print('When K=3, the accuracy of VGG16 is %.2f %%, the accuracy of AlexNet is %.2f %%.' % (
        vgg16_accuracy_3, alex_accuracy_3))
    print('When K=5, the accuracy of VGG16 is %.2f %%, the accuracy of AlexNet is %.2f %%.' % (
        vgg16_accuracy_5, alex_accuracy_5))


if __name__ == "__main__":
    # In 'main()', it saves all extracted date to .mat file.
    # No need to run again if all files exist
    # main()
    KNN_test()
