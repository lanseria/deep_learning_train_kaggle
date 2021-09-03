# 首先导入包
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2 as cv
import os
import matplotlib.pyplot as plt
import torchvision.models as models
# This is for the progress bar.
from tqdm import tqdm
import seaborn as sns

DATA_DIR_NAME = 'classify-leaves'
TRAIN_CSV_PATH = os.path.join("data", DATA_DIR_NAME, 'train.csv')
TEST_CSV_PATH = os.path.join("data", DATA_DIR_NAME, 'test.csv')
IMG_PATH = os.path.join("data", DATA_DIR_NAME)
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

# a = range(10)
# b = map(lambda i, x: i, x*x, enumerate(a))
# https://www.kaggle.com/nekokiku/simple-resnet-baseline
# print(list(enumerate(a)))

# 继承pytorch的dataset，创建自己的


class LeavesData(Dataset):
    def __init__(self, csv_path, file_path, mode='train', valid_ratio=0.2, resize_height=256, resize_width=256):
        """
        Args:
            csv_path (string): csv 文件路径
            file_path (string): 图像文件所在路径
            mode (string): 训练模式还是测试模式
            valid_ratio (float): 验证集比例
        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.mode = mode

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(
            csv_path, header=None)  # header=None是去掉表头部分
        # 计算 length
        self.data_len = len(self.data_info.index) - 1
        self.train_len = int(self.data_len * (1 - valid_ratio))

        if mode == 'train':
            # 第一列包含图像文件的名称
            # self.data_info.iloc[1:,0]表示读取第一列，从第二行开始到train_len
            self.train_image = np.asarray(
                self.data_info.iloc[1:self.train_len, 0])
            # 第二列是图像的 label
            self.train_label = np.asarray(
                self.data_info.iloc[1:self.train_len, 1])
            self.image_arr = self.train_image
            self.label_arr = self.train_label
        elif mode == 'valid':
            self.valid_image = np.asarray(
                self.data_info.iloc[self.train_len:, 0])
            self.valid_label = np.asarray(
                self.data_info.iloc[self.train_len:, 1])
            self.image_arr = self.valid_image
            self.label_arr = self.valid_label
        elif mode == 'test':
            self.test_image = np.asarray(self.data_info.iloc[1:, 0])
            self.image_arr = self.test_image

        self.real_len = len(self.image_arr)

        print('Finished reading the {} set of Leaves Dataset ({} samples found)'
              .format(mode, self.real_len))

    def __getitem__(self, index):
        # 从 image_arr中得到索引对应的文件名
        single_image_name = self.image_arr[index]

        # 读取图像文件
        # img_as_img = cv.imread(os.path.join(
        #     self.file_path, single_image_name))
        img_as_img = Image.open(os.path.join(
            self.file_path, single_image_name))

        # 如果需要将RGB三通道的图片转换成灰度图片可参考下面两行
        # if img_as_img.mode != 'L':
        #     img_as_img = img_as_img.convert('L')

        # 设置好需要转换的变量，还可以包括一系列的nomarlize等等操作
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
                transforms.ToTensor()
            ])
        else:
            # valid和test不做数据增强
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

        img_as_img = transform(img_as_img)

        if self.mode == 'test':
            return img_as_img
        else:
            # 得到图像的 string label
            label = self.label_arr[index]
            # number label
            number_label = class_to_num[label]

            return img_as_img, number_label  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.real_len


def run_dataset():
    train_dataset = LeavesData(TRAIN_CSV_PATH, IMG_PATH, mode='train')
    val_dataset = LeavesData(TRAIN_CSV_PATH, IMG_PATH, mode='valid')
    test_dataset = LeavesData(TEST_CSV_PATH, IMG_PATH, mode='test')
    # print(train_dataset)
    # print(val_dataset)
    # print(test_dataset)
    # img, lab = train_dataset[0]
    # print(img.shape, lab)
    # print(val_dataset[0])
    # print(test_dataset[0])

    # print(len(train_dataset))
    # print(len(val_dataset))
    # print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=5
    )
    return train_loader, val_loader, test_loader


# 给大家展示一下数据长啥样

def im_convert(tensor):
    """ 展示数据"""
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)

    return image


if __name__ == '__main__':
    train_loader, val_loader, test_loader = run_dataset()
    fig = plt.figure(figsize=(20, 12))
    columns = 4
    rows = 2

    dataiter = iter(val_loader)
    inputs, classes = dataiter.next()

    for idx in range(columns*rows):
        ax = fig.add_subplot(rows, columns, idx+1, xticks=[], yticks=[])
        ax.set_title(num_to_class[int(classes[idx])])
        plt.imshow(im_convert(inputs[idx]))

    plt.show()
