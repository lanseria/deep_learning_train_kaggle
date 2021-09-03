import torch_utils as tu
import os
import d2l
from d2l.torch import Residual
import numpy as np
import pandas as pd
import torch
from torch import nn
import cv2 as cv
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedKFold
import torch_utils as tu

SEED = 20
tu.tools.seed_everything(SEED, deterministic=False)

DATA_DIR_NAME = 'classify-leaves'
CLASSES = 176
FOLD = 5
BATCH_SIZE = 128
ACCUMULATE = 1
LR = 0.01
EPOCH = 50
DECAY_SCALE = 20.0
MIXUP = 0
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
    # print(get_indexs_by_labels(['maclura_pomifera'])[0])
    for item in label_lists:
        csv_data.loc[csv_data['label'] == item, "label"] = get_indexs_by_labels([item])[
            0]
    csv_fname = os.path.join("data", DATA_DIR_NAME, 'train_number.csv')
    csv_data.to_csv(csv_fname)  # 绝对位置


[label_lists, label_sizes] = set_train_label_lists()
# print(label_lists)
# test_id = get_indexs_by_labels(['maclura_pomifera'])
# print(test_id)
# set_train_csv_number()

csv_fname = os.path.join("data", DATA_DIR_NAME, 'train_number.csv')
train_df = pd.read_csv(csv_fname)
sfolder = StratifiedKFold(n_splits=FOLD, random_state=SEED, shuffle=True)
tr_folds = []
val_folds = []
for train_idx, val_idx in sfolder.split(train_df.image, train_df.label):
    tr_folds.append(train_idx)
    val_folds.append(val_idx)
    print(train_idx, val_idx)

train_iter = torch.utils.data.DataLoader(datasets.ImageFolder(
    os.path.join("data", DATA_DIR_NAME, 'images')), batch_size=BATCH_SIZE, shuffle=True)

print(train_iter)
