# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

DATA_DIR_NAME = 'classify-leaves'
TRAIN_CSV_PATH = os.path.join("data", DATA_DIR_NAME, 'train.csv')
# 看看label文件长啥样
labels_dataframe = pd.read_csv(TRAIN_CSV_PATH)
# print(labels_dataframe.head(5))
# print(labels_dataframe.describe())

# function to show bar length


# def barw(ax):

#     for p in ax.patches:
#         val = p.get_width()  # height of the bar
#         x = p.get_x() + p.get_width()  # x- position
#         y = p.get_y() + p.get_height()/2  # y-position
#         ax.annotate(round(val, 2), (x, y))

# # finding top leaves


# plt.figure(figsize=(15, 30))
# ax0 = sns.countplot(
#     y=labels_dataframe['label'], order=labels_dataframe['label'].value_counts().index)
# barw(ax0)
# plt.show()

# 把label文件排个序
leaves_labels = sorted(list(set(labels_dataframe['label'])))
n_classes = len(leaves_labels)
# print(n_classes)
# leaves_labels[:10]
# 把label转成对应的数字
class_to_num = dict(zip(leaves_labels, range(n_classes)))
# print(class_to_num)
# 再转换回来，方便最后预测的时候使用
num_to_class = {v: k for k, v in class_to_num.items()}

a = range(10)
# b = map(lambda i, x: i, x*x, enumerate(a))
# https://www.kaggle.com/nekokiku/simple-resnet-baseline
print(list(enumerate(a)))
