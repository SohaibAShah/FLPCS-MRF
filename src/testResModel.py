import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import time
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import random
import csv
from datetime import datetime
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys


# Define dataset loader
class CostumDataset(Dataset):
    def __init__(self, features1, features2, features3, labels):
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.labels = labels

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, index):
        return self.features1[index], self.features2[index], self.features3[index], self.labels[index]

def scaled_data(X_train, X_test, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled

def scaled_data(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled

def set_seed(seed=0):
    # 设置环境变量以确保可重复性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 设置 Python 随机种子
    random.seed(seed)
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有 GPU 进行设置
    # 确保所有操作都在单线程中进行（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loadClientsData():
    subs = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    X_train_csv_scaled_splits = {}
    X_test_csv_scaled_splits = {}
    Y_train_csv_splits = {}
    Y_test_csv_splits = {}
    X_train_1_scaled_splits = {}
    X_test_1_scaled_splits = {}
    Y_train_1_splits = {}
    Y_test_1_splits = {}
    X_train_2_scaled_splits = {}
    X_test_2_scaled_splits = {}
    Y_train_2_splits = {}
    Y_test_2_splits = {}
    clint_index = 0
    for sub_ in subs:
        SUB_train = pd.read_csv('./dataset/Sensor + Image/{}_sensor_train.csv'.format(sub_), skiprows=1)
        SUB_train.head()
        # print('{}_SUB.shap'.format(sub_),SUB_train.shape)

        SUB_train.isnull().sum()
        NA_cols = SUB_train.columns[SUB_train.isnull().any()]
        # print('Columns contain NULL values : \n', NA_cols)
        SUB_train.dropna(inplace=True)
        SUB_train.drop_duplicates(inplace=True)
        # print('Sensor Data shape after dropping NaN and redudant samples :', SUB_train.shape)
        times_train = SUB_train['Time']
        list_DROP = ['Infrared 1',
                     'Infrared 2',
                     'Infrared 3',
                     'Infrared 4',
                     'Infrared 5',
                     'Infrared 6']
        SUB_train.drop(list_DROP, axis=1, inplace=True)
        SUB_train.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

        # print('{}_train_Sensor Data shape after dropping columns contain NaN values :'.format(sub_), SUB_train.shape)

        SUB_train.set_index('Time', inplace=True)
        SUB_train.head()

        cam = '1'
        image_train = './dataset/Sensor + Image' + '/' + '{}_image_1_train.npy'.format(sub_)
        name_train = './dataset/Sensor + Image' + '/' + '{}_name_1_train.npy'.format(sub_)
        label_train = './dataset/Sensor + Image' + '/' + '{}_label_1_train.npy'.format(sub_)

        img_1_train = np.load(image_train)
        label_1_train = np.load(label_train)
        name_1_train = np.load(name_train)

        cam = '2'

        image_train = './dataset/Sensor + Image' + '/' + '{}_image_2_train.npy'.format(sub_)
        name_train = './dataset/Sensor + Image' + '/' + '{}_name_2_train.npy'.format(sub_)
        label_train = './dataset/Sensor + Image' + '/' + '{}_label_2_train.npy'.format(sub_)

        img_2_train = np.load(image_train)
        label_2_train = np.load(label_train)
        name_2_train = np.load(name_train)


        # print('{}_len(img_1_train)'.format(sub_),len(img_1_train))
        # print('{}_len(name_1_train)'.format(sub_),len(name_1_train))
        # print('{}_len(img_2_train)'.format(sub_),len(img_2_train))
        # print('{}_len(name_2_train)'.format(sub_),len(name_2_train))

        # remove NaN values corresponding to index sample in csv file
        redundant_1 = list(set(name_1_train) - set(times_train))
        redundant_2 = list(set(name_2_train) - set(times_train))
        # ind = np.arange(0, 294677)
        ind = np.arange(0, len(img_1_train))

        red_in1 = ind[np.isin(name_1_train, redundant_1)]
        name_1_train = np.delete(name_1_train, red_in1)
        img_1_train = np.delete(img_1_train, red_in1, axis=0)
        label_1_train = np.delete(label_1_train, red_in1)

        red_in2 = ind[np.isin(name_2_train, redundant_2)]
        name_2_train = np.delete(name_2_train, red_in2)
        img_2_train = np.delete(img_2_train, red_in2, axis=0)
        label_2_train = np.delete(label_2_train, red_in2)

        # print('{}_len(name_1_train)'.format(sub_),len(name_1_train))
        # print('{}_len(name_2_train)'.format(sub_),len(name_2_train))

        class_name = ['?????',
                      'Falling hands',
                      'Falling knees',
                      'Falling backwards',
                      'Falling sideward',
                      ' Falling chair',
                      ' Walking',
                      'Standing',
                      'Sitting',
                      'Picking object',
                      'Jumping',
                      'Laying']


        data_train = SUB_train.loc[name_1_train].values
        # print(img_1_train.shape)
        # print(img_2_train.shape)
        # print(data_train.shape)

        # print((label_2_train == data_train[:, -1]).all())
        # print((label_1_train == data_train[:, -1]).all())

        set_seed()
        X_csv_train, y_csv_train = data_train[:, :-1], data_train[:, -1]
        y_csv_train = np.where(y_csv_train == 20, 0, y_csv_train)

        label_1_train = np.where(label_1_train == 20, 0, label_1_train)
        label_2_train = np.where(label_2_train == 20, 0, label_2_train)

        # print('X_csv_train shape : ', X_csv_train.shape)
        # print('y_csv_train shape : ', y_csv_train.shape)

        Y_csv_train = torch.nn.functional.one_hot(torch.from_numpy(y_csv_train).long(), 12).float()

        X_csv_train_scaled = scaled_data(X_csv_train)

        # print('X_csv_train_scaled shape : ', X_csv_train_scaled.shape)
        # print('Y_csv_train shape : ', Y_csv_train.shape)

        X_train_1 = img_1_train
        y_train_1 = label_1_train

        # print('*' * 20)
        # print('X_train_1 shape : ', X_train_1.shape)
        # print('y_train_1 shape : ', y_train_1.shape)

        Y_train_1 = torch.nn.functional.one_hot(torch.from_numpy(y_train_1).long(), 12).float()

        # print('X_train_1 shape : ', X_train_1.shape)
        # print('y_train_1 shape : ', Y_train_1.shape)

        X_train_2 = img_2_train
        y_train_2 = label_2_train

        # print('*' * 20)
        # print('X_train_2 shape : ', X_train_2.shape)
        # print('y_train_2 shape : ', y_train_2.shape)

        Y_train_2 = torch.nn.functional.one_hot(torch.from_numpy(y_train_2).long(), 12).float()

        # print('X_train_2 shape : ', X_train_2.shape)
        # print('y_train_2 shape : ', Y_train_2.shape)


        # print('(y_train_1 == y_csv_train).all():',(y_train_1 == y_csv_train).all())
        # print('(y_train_2 == y_csv_train).all()',(y_train_2 == y_csv_train).all())


        shape1, shape2 = 32, 32
        X_train_1 = X_train_1.reshape(X_train_1.shape[0], shape1, shape2, 1)
        X_train_2 = X_train_2.reshape(X_train_2.shape[0], shape1, shape2, 1)


        X_train_1_scaled = X_train_1 / 255.0
        X_train_2_scaled = X_train_2 / 255.0

        SUB_test = pd.read_csv('./dataset/Sensor + Image/{}_sensor_test.csv'.format(sub_), skiprows=1)
        SUB_test.head()
        # print('{}_SUB.shap'.format(sub_), SUB_test.shape)

        SUB_test.isnull().sum()
        NA_cols = SUB_test.columns[SUB_test.isnull().any()]
        # print('Columns contain NULL values : \n', NA_cols)
        SUB_test.dropna(inplace=True)
        SUB_test.drop_duplicates(inplace=True)
        # print('Sensor Data shape after dropping NaN and redudant samples :', SUB_test.shape)
        times_test = SUB_test['Time']
        list_DROP = ['Infrared 1',
                     'Infrared 2',
                     'Infrared 3',
                     'Infrared 4',
                     'Infrared 5',
                     'Infrared 6']
        SUB_test.drop(list_DROP, axis=1, inplace=True)
        SUB_test.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

        # print('{}_test_Sensor Data shape after dropping columns contain NaN values :'.format(sub_), SUB_test.shape)

        SUB_test.set_index('Time', inplace=True)
        SUB_test.head()

        cam = '1'
        image_test = './dataset/Sensor + Image' + '/' + '{}_image_1_test.npy'.format(sub_)
        name_test = './dataset/Sensor + Image' + '/' + '{}_name_1_test.npy'.format(sub_)
        label_test = './dataset/Sensor + Image' + '/' + '{}_label_1_test.npy'.format(sub_)

        img_1_test = np.load(image_test)
        label_1_test = np.load(label_test)
        name_1_test = np.load(name_test)

        cam = '2'

        image_test = './dataset/Sensor + Image' + '/' + '{}_image_2_test.npy'.format(sub_)
        name_test = './dataset/Sensor + Image' + '/' + '{}_name_2_test.npy'.format(sub_)
        label_test = './dataset/Sensor + Image' + '/' + '{}_label_2_test.npy'.format(sub_)

        img_2_test = np.load(image_test)
        label_2_test = np.load(label_test)
        name_2_test = np.load(name_test)


        # print('{}_len(img_1_test)'.format(sub_), len(img_1_test))
        # print('{}_len(name_1_test)'.format(sub_), len(name_1_test))
        # print('{}_len(img_2_test)'.format(sub_), len(img_2_test))
        # print('{}_len(name_2_test)'.format(sub_), len(name_2_test))

        # remove NaN values corresponding to index sample in csv file
        redundant_1 = list(set(name_1_test) - set(times_test))
        redundant_2 = list(set(name_2_test) - set(times_test))
        # ind = np.arange(0, 294677)
        ind = np.arange(0, len(img_1_test))

        red_in1 = ind[np.isin(name_1_test, redundant_1)]
        name_1_test = np.delete(name_1_test, red_in1)
        img_1_test = np.delete(img_1_test, red_in1, axis=0)
        label_1_test = np.delete(label_1_test, red_in1)

        red_in2 = ind[np.isin(name_2_test, redundant_2)]
        name_2_test = np.delete(name_2_test, red_in2)
        img_2_test = np.delete(img_2_test, red_in2, axis=0)
        label_2_test = np.delete(label_2_test, red_in2)

        # print('{}_len(name_1_test)'.format(sub_), len(name_1_test))
        # print('{}_len(name_2_test)'.format(sub_), len(name_2_test))

        class_name = ['?????',
                      'Falling hands',
                      'Falling knees',
                      'Falling backwards',
                      'Falling sideward',
                      ' Falling chair',
                      ' Walking',
                      'Standing',
                      'Sitting',
                      'Picking object',
                      'Jumping',
                      'Laying']

        data_test = SUB_test.loc[name_1_test].values
        # print(img_1_test.shape)
        # print(img_2_test.shape)
        # print(data_test.shape)

        # print((label_2_test == data_test[:, -1]).all())
        # print((label_1_test == data_test[:, -1]).all())

        set_seed()
        X_csv_test, y_csv_test = data_test[:, :-1], data_test[:, -1]
        y_csv_test = np.where(y_csv_test == 20, 0, y_csv_test)

        label_1_test = np.where(label_1_test == 20, 0, label_1_test)
        label_2_test = np.where(label_2_test == 20, 0, label_2_test)

        # print('X_csv_test shape : ', X_csv_test.shape)
        # print('y_csv_test shape : ', y_csv_test.shape)

        Y_csv_test = torch.nn.functional.one_hot(torch.from_numpy(y_csv_test).long(), 12).float()

        X_csv_test_scaled = scaled_data(X_csv_test)

        # print('X_csv_test_scaled shape : ', X_csv_test_scaled.shape)
        # print('Y_csv_test shape : ', Y_csv_test.shape)

        X_test_1 = img_1_test
        y_test_1 = label_1_test

        # print('*' * 20)
        # print('X_test_1 shape : ', X_test_1.shape)
        # print('y_test_1 shape : ', y_test_1.shape)

        Y_test_1 = torch.nn.functional.one_hot(torch.from_numpy(y_test_1).long(), 12).float()

        # print('X_test_1 shape : ', X_test_1.shape)
        # print('y_test_1 shape : ', Y_test_1.shape)

        X_test_2 = img_2_test
        y_test_2 = label_2_test

        # print('*' * 20)
        # print('X_test_2 shape : ', X_test_2.shape)
        # print('y_test_2 shape : ', y_test_2.shape)

        Y_test_2 = torch.nn.functional.one_hot(torch.from_numpy(y_test_2).long(), 12).float()

        # print('X_test_2 shape : ', X_test_2.shape)
        # print('y_test_2 shape : ', Y_test_2.shape)

        # print('(y_test_1 == y_csv_test).all():', (y_test_1 == y_csv_test).all())
        # print('(y_test_2 == y_csv_test).all()', (y_test_2 == y_csv_test).all())


        X_test_1 = X_test_1.reshape(X_test_1.shape[0], shape1, shape2, 1)
        X_test_2 = X_test_2.reshape(X_test_2.shape[0], shape1, shape2, 1)

        X_test_1_scaled = X_test_1 / 255.0
        X_test_2_scaled = X_test_2 / 255.0

        # print(X_train_1_scaled.shape)
        # print(X_test_1_scaled.shape)
        #
        # print(X_train_2_scaled.shape)
        # print(X_test_2_scaled.shape)

        X_train_csv_scaled_splits[clint_index] = X_csv_train_scaled
        X_test_csv_scaled_splits[clint_index] = X_csv_test_scaled
        Y_train_csv_splits[clint_index] = Y_csv_train
        Y_test_csv_splits[clint_index] = Y_csv_test
        X_train_1_scaled_splits[clint_index] = X_train_1_scaled
        X_test_1_scaled_splits[clint_index] = X_test_1_scaled
        Y_train_1_splits[clint_index] = Y_train_1
        Y_test_1_splits[clint_index] = Y_test_1
        X_train_2_scaled_splits[clint_index] = X_train_2
        X_test_2_scaled_splits[clint_index] = X_test_2_scaled
        Y_train_2_splits[clint_index] = Y_train_2
        Y_test_2_splits[clint_index] = Y_test_2
        clint_index += 1
    return X_train_csv_scaled_splits,X_test_csv_scaled_splits, Y_train_csv_splits,Y_test_csv_splits,X_train_1_scaled_splits,X_test_1_scaled_splits,Y_train_1_splits,Y_test_1_splits,X_train_2_scaled_splits,X_test_2_scaled_splits,Y_train_2_splits,Y_test_2_splits


# 2. 定义模型
class SimpleModel1(nn.Module):

    def __init__(self, num_csv_features, img_shape1, img_shape2,catType):
        super(SimpleModel1, self).__init__()
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)
        self.catType = catType

        # 第二输入分支：处理第一张图像的2D卷积
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # 第三输入分支：处理第二张图像的2D卷积
        self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img2_batch_norm = nn.BatchNorm2d(18)
        self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img2_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img2_dropout = nn.Dropout(0.2)

        # 全连接层
        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1200, 800)
        self.dr2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1600, 12)

    def forward(self, x_csv, x_img1, x_img2):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)
        # x_csv = self.fc_csv_3(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        # 第二分支：第一张图像
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.contiguous().view(x_img1.size(0), -1)
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 = self.img1_dropout(x_img1)

        x_img2 = x_img2.permute(0, 3, 1, 2)
        # 第三分支：第二张图像
        x_img2 = F.relu(self.img2_conv(x_img2))
        x_img2 = self.img2_batch_norm(x_img2)
        x_img2 = self.img2_pool(x_img2)
        x_img2 = x_img2.contiguous().view(x_img2.size(0), -1)
        x_img2 = F.relu(self.img2_fc1(x_img2))
        x_img2 = self.img2_dropout(x_img2)

        # 连接三个分支

        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        # x += residual
        # Element-wise Addition,Column-wise concatenation,Element-wise Multiplication，Element-wise Max/Min
        if self.catType =='Element-wise Addition':
            x = residual+ x
        elif self.catType =='Column-wise concatenation':
            x = torch.cat((residual, x), dim=1)
        elif self.catType =='Element-wise Multiplication':
            x = residual* x
        elif self.catType =='Element-wise Max':
            x = torch.max(residual, x)
        elif self.catType =='Element-wise Min':
            x = torch.min(residual, x)
        x = F.softmax(self.fc3(x), dim=1)

        return x


class SimpleModel2(nn.Module):

    def __init__(self, num_csv_features, img_shape1, img_shape2):
        super(SimpleModel2, self).__init__()
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)

        # 第二输入分支：处理第一张图像的2D卷积
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # 第三输入分支：处理第二张图像的2D卷积
        self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img2_batch_norm = nn.BatchNorm2d(18)
        self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img2_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img2_dropout = nn.Dropout(0.2)

        # 全连接层
        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1200, 800)
        self.dr2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(2000, 12)

    def forward(self, x_csv, x_img1, x_img2):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)
        # x_csv = self.fc_csv_3(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        # 第二分支：第一张图像
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.contiguous().view(x_img1.size(0), -1)
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 = self.img1_dropout(x_img1)

        x_img2 = x_img2.permute(0, 3, 1, 2)
        # 第三分支：第二张图像
        x_img2 = F.relu(self.img2_conv(x_img2))
        x_img2 = self.img2_batch_norm(x_img2)
        x_img2 = self.img2_pool(x_img2)
        x_img2 = x_img2.contiguous().view(x_img2.size(0), -1)
        x_img2 = F.relu(self.img2_fc1(x_img2))
        x_img2 = self.img2_dropout(x_img2)

        # 连接三个分支

        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = F.relu(self.fc2(x))
        x = self.dr2(x)
        # x += residual
        x = torch.cat((residual, x), dim=1)
        x = F.softmax(self.fc3(x), dim=1)

        return x

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_id, batch in enumerate(train_loader):
        data1 = batch[0].to(device).float()
        data2 = batch[1].to(device).float()
        data3 = batch[2].to(device).float()
        target = torch.squeeze(batch[3]).to(device).float()

        output = model(data1, data2, data3)
        loss = criterion(output, target)

        # pred = output.detach().max(1)[1]
        # target_ =  target.detach().max(1)[1]
        # correct += pred.eq(target.detach().max(1)[1]).sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * data1.size()[0]
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.max(1)[1]).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc



def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            data1 = batch[0].to(device).float()
            data2 = batch[1].to(device).float()
            data3 = batch[2].to(device).float()
            target = torch.squeeze(batch[3]).to(device).float()

            output = model(data1, data2, data3)
            loss = criterion(output, target)

            # 统计
            running_loss += loss.item() * data1.size()[0]
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.max(1)[1]).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc



# load data
X_train_csv_scaled_splits,X_test_csv_scaled_splits, \
    Y_train_csv_splits,Y_test_csv_splits,\
    X_train_1_scaled_splits,X_test_1_scaled_splits,\
    Y_train_1_splits,Y_test_1_splits,\
    X_train_2_scaled_splits,X_test_2_scaled_splits,\
    Y_train_2_splits,Y_test_2_splits = loadClientsData()
# catTypes = {'Column-wise concatenation','Element-wise Addition','Element-wise Multiplication','Element-wise Max','Element-wise Min'}
catTypes = {'Column-wise concatenation'}
for catType in catTypes:
    print('catType',catType)
    start_time = time.time()
    for clinet_index in range(len(X_train_csv_scaled_splits)):
        # print(X_train_csv_scaled_splits[clinet_index].shape)
        train_dataset = CostumDataset(X_train_csv_scaled_splits[clinet_index],X_train_1_scaled_splits[clinet_index],X_train_2_scaled_splits[clinet_index],Y_train_csv_splits[clinet_index])
        val_dataset = CostumDataset(X_test_csv_scaled_splits[clinet_index],X_test_1_scaled_splits[clinet_index],X_test_2_scaled_splits[clinet_index],Y_test_csv_splits[clinet_index])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = SimpleModel1(X_train_csv_scaled_splits[0].shape[1],32,32,catType)

        # 3. 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 4. 训练和验证函数

        # 5. 训练主循环
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        num_epochs = 50
        best_acc = 0.0

        for epoch in range(num_epochs):


            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "client_{}_catType_{}_best_model.pth".format(clinet_index,catType))



            # 打印每个epoch的结果
            # print(f"Epoch {epoch+1}/{num_epochs} - "
            #       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            #       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
            #       f"Time: {end_time - start_time:.2f}s")

        # print("{:.2f}%".format(clinet_index,best_acc))
        print("{:.2f}%".format(best_acc))
    end_time = time.time()
    print(f"Time: {end_time - start_time:.2f}s")
