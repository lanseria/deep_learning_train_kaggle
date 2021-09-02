import os
import d2l
from d2l.torch import Residual
import numpy as np
import pandas as pd
import torch
from torch import nn
import cv2 as cv
import torchvision.transforms as transforms

DATA_DIR_NAME = 'classify-leaves'
label_lists = []
label_sizes = 0


def read_data(is_train=True):
    csv_fname = os.path.join("data", DATA_DIR_NAME,
                             'train.csv' if is_train else 'test.csv')
    csv_data = pd.read_csv(csv_fname)
    return csv_data


def set_train_label_lists():
    csv_data = read_data(True).set_index('image')
    label_sets = set()
    if os.path.exists('label_lists.npy'):
        print('label_lists.npy 文件存在')
        label_lists = np.load('label_lists.npy').tolist()
        # print(label_lists)
    else:
        print('label_lists.npy 文件不存在')
        for image, target in csv_data.iterrows():
            # print('image'+image)
            label_sets.add(target.values[0])
        label_lists = list(label_sets)
        label_lists = np.array(label_lists)
        np.save('label_lists.npy', label_lists)
    # print(label_lists)
    label_sizes = len(label_lists)
    return [label_lists, label_sizes]


def get_labels_by_indexs(indexs):
    return [label_lists[int(i)] for i in indexs]


def get_indexs_by_labels(labels):
    return [label_lists.index(l) for l in labels]


def set_train_csv_number():
    csv_data = read_data(True)
    # csv_data.loc['maclura_pomifera']
    # for image, target in csv_data.iterrows():
    #     target.values = get_indexs_by_labels(target.values)

    csv_data.loc[csv_data['label'] == 'maclura_pomifera']['label'] = get_indexs_by_labels([
                                                                                          'maclura_pomifera'])[0]
    print(csv_data)


[label_lists, label_sizes] = set_train_label_lists()
# print(label_lists)
# test_id = get_indexs_by_labels(['maclura_pomifera'])
# print(test_id)
set_train_csv_number()