import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
import cv2


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def cifar_10_data(dir):
    label2name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    train_x = []
    train_y = []
    filename = dir + r'\data_batch_'
    for i in range(1, 6):
        cur_file = filename + str(i)
        dicts = unpickle(cur_file)
        images = dicts[b'data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1]).astype(np.float32) / 255.0
        train_x += list(images)
        labels = np.array(dicts[b'labels'])
        train_y += list(labels)
    test_file = dir + r'\test_batch'
    dicts = unpickle(test_file)
    images = dicts[b'data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1]).astype(np.float32) / 255.0
    labels = dicts[b'labels']
    test_x = list(images)
    test_y = list(labels)
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y), label2name


def getbatch(train_x, train_y, batch_size, shuffle=True):
    list = np.arange(train_x.shape[0])
    if shuffle == True:
        np.random.shuffle(list)
        start = 0
        for step in range(0, math.ceil(train_x.shape[0] / batch_size)):
            one_batch = list[start: start + batch_size]
            start += batch_size
            out_x = train_x[one_batch]
            out_y = train_y[one_batch]
            yield out_x, out_y
    else:
        start = 0
        for step in range(0, math.ceil(train_x.shape[0] / batch_size)):
            one_batch = list[start: start + batch_size]
            start += batch_size
            out_x = train_x[one_batch]
            out_y = train_y[one_batch]
            yield out_x, out_y



# dicts = unpickle(r'D:\cifar\cifar-10-python\cifar-10-batches-py\test_batch')
# images = dicts[b'data'].reshape([10000, 3, 32, 32]).transpose([0, 2, 3, 1]).astype(np.float32)/255.0
# print(dicts[b'labels'])
# plt.imshow(images[4])
# plt.show()
