import d2l
from d2l.torch import Residual
import pandas as pd
import torch
from torch import nn
import cv2 as cv
import torchvision.transforms as transforms

DATA_DIR_NAME = 'classify-leaves'

train_csv = pd.read_csv("./data/" + DATA_DIR_NAME + '/train.csv')
# test_csv = pd.read_csv("./data/" + DATA_DIR_NAME + '/test.csv')

# print(train_csv.shape)


train_features_images = train_csv.image
train_labels = train_csv.label

# print(train_features_images, train_labels)

# valid_csv
valid_csv = train_csv.sample(800)

valid_features_images = valid_csv.image
valid_labels = valid_csv.label
# print(valid_csv)
# print(valid_features_images, valid_labels)

valid_features = []


def stack(item):
    img = cv.imread("./data/" + DATA_DIR_NAME + "/" + item)
    # print(img.shape)
    transToTensor = transforms.ToTensor()

    img_tensor = transToTensor(img)  # tensor数据格式是torch(C,H,W)
    # print(img_tensor.size())
    valid_features.append(img_tensor)


for column_name, item in valid_features_images.iteritems():
    stack(item)

valid_features = torch.stack(valid_features, dim=0)
print(valid_features.shape)

b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 176))


lr, num_epochs, batch_size = 0.05, 10, 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# X = torch.rand(size=(1, 3, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape:\t', X.shape)
