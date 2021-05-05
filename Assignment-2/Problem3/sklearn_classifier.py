from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
import numpy as np
import time


if __name__ == "__main__":
    train_vgg_mat = sio.loadmat('vgg16_train.mat')
    train_vgg_feature = np.asarray(train_vgg_mat['feature'])
    train_vgg_label = train_vgg_mat['label'][0]

    train_alex_mat = sio.loadmat('alexnet_train.mat')
    train_alex_feature = np.asarray(train_alex_mat['feature'])
    train_alex_label = train_alex_mat['label'][0]

    test_vgg_mat = sio.loadmat('vgg16_test.mat')
    test_vgg_feature = np.asarray(test_vgg_mat['feature'])
    test_vgg_label = test_vgg_mat['label'][0]

    test_alex_mat = sio.loadmat('alexnet_test.mat')
    test_alex_feature = np.asarray(test_alex_mat['feature'])
    test_alex_label = test_alex_mat['label'][0]

    dataset = [
        [train_vgg_feature, train_vgg_label,
            test_vgg_feature[:1000], test_vgg_label[:1000]],
        [train_alex_feature, train_alex_label,
            test_alex_feature[:1000], test_alex_label[:1000]]
    ]

    train_net = ['VGG16', 'AlexNet']
    for idx, data in enumerate(dataset):
        train_set, train_label, test_set, test_label = data
        K_num = [1, 3, 5]
        print('For %s feature' % train_net[idx])
        for K in K_num:
            KNN_clf = KNeighborsClassifier(n_neighbors=K, weights='uniform')

            KNN_clf.fit(train_set, train_label)

            acc = KNN_clf.score(test_set, test_label)

            print('When K=%d, accuracy=%.2f %%' % (K, acc*100))
