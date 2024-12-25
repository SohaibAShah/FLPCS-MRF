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


def display_result(y_test, y_pred):
    print('Accuracy score : ', accuracy_score(y_test, y_pred))
    print('Precision score : ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall score : ', recall_score(y_test, y_pred, average='weighted'))
    print('F1 score : ', f1_score(y_test, y_pred, average='weighted'))


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



def loadData():
    SUB = pd.read_csv('./dataset/Sensor + Image/sensor.csv', skiprows=1)
    SUB.head()
    print(SUB.shape)

    SUB.isnull().sum()
    NA_cols = SUB.columns[SUB.isnull().any()]
    print('Columns contain NULL values : \n', NA_cols)
    SUB.dropna(inplace=True)
    SUB.drop_duplicates(inplace=True)
    print('Sensor Data shape after dropping NaN and redudant samples :', SUB.shape)
    times = SUB['Time']
    list_DROP = ['Infrared 1',
                 'Infrared 2',
                 'Infrared 3',
                 'Infrared 4',
                 'Infrared 5',
                 'Infrared 6']
    SUB.drop(list_DROP, axis=1, inplace=True)
    SUB.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

    print('Sensor Data shape after dropping columns contain NaN values :', SUB.shape)

    SUB.set_index('Time', inplace=True)
    SUB.head()

    cam = '1'

    image = './dataset/Sensor + Image' + '/' + 'image_' + cam + '.npy'
    name = './dataset/Sensor + Image' + '/' + 'name_' + cam + '.npy'
    label = './dataset/Sensor + Image' + '/' + 'label_' + cam + '.npy'

    img_1 = np.load(image)
    label_1 = np.load(label)
    name_1 = np.load(name)

    cam = '2'

    image = './dataset/Sensor + Image' + '/' + 'image_' + cam + '.npy'
    name = './dataset/Sensor + Image' + '/' + 'name_' + cam + '.npy'
    label = './dataset/Sensor + Image' + '/' + 'label_' + cam + '.npy'

    img_2 = np.load(image)
    label_2 = np.load(label)
    name_2 = np.load(name)


    print(len(img_1))
    print(len(name_1))
    print(len(img_2))
    print(len(name_2))


    # remove NaN values corresponding to index sample in csv file
    redundant_1 = list(set(name_1) - set(times))
    redundant_2 = list(set(name_2) - set(times))
    # ind = np.arange(0, 294677)
    ind = np.arange(0, len(img_1))

    red_in1 = ind[np.isin(name_1, redundant_1)]
    name_1 = np.delete(name_1, red_in1)
    img_1 = np.delete(img_1, red_in1, axis=0)
    label_1 = np.delete(label_1, red_in1)

    red_in2 = ind[np.isin(name_2, redundant_2)]
    name_2 = np.delete(name_2, red_in2)
    img_2 = np.delete(img_2, red_in2, axis=0)
    label_2 = np.delete(label_2, red_in2)


    print(len(name_2))
    print(len(name_1))

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

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_1[i], cmap='gray')
        plt.xlabel(class_name[label_1[i]])
    plt.show()


    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_2[i], cmap='gray')
        plt.xlabel(class_name[label_2[i]])
    plt.show()


    data = SUB.loc[name_1].values
    print(img_1.shape)
    print(img_2.shape)
    print(data.shape)

    print((label_2 == data[:, -1]).all())
    print((label_1 == data[:, -1]).all())

    set_seed()
    X_csv, y_csv = data[:, :-1], data[:, -1]

    y_csv = np.where(y_csv == 20, 0, y_csv)
    label_1 = np.where(label_1 == 20, 0, label_1)
    label_2 = np.where(label_2 == 20, 0, label_2)
    X_train_csv, X_rem_csv, y_train_csv, y_rem_csv = train_test_split(X_csv, y_csv,
                                                                      train_size=0.6,
                                                                      random_state=42)

    X_val_csv, X_test_csv, y_val_csv, y_test_csv = train_test_split(X_rem_csv, y_rem_csv,
                                                                    test_size=0.5,
                                                                    random_state=42)

    print('X_train_csv shape : ', X_train_csv.shape)
    print('X_test_csv shape : ', X_test_csv.shape)
    print('X_val_csv shape : ', X_val_csv.shape)
    print('y_train_csv shape : ', y_train_csv.shape)
    print('y_test_csv shape : ', y_test_csv.shape)
    print('y_val_csv shape : ', y_val_csv.shape)

    Y_train_csv = torch.nn.functional.one_hot(torch.from_numpy(y_train_csv).long(), 12).float()
    Y_test_csv = torch.nn.functional.one_hot(torch.from_numpy(y_test_csv).long(), 12).float()
    Y_val_csv = torch.nn.functional.one_hot(torch.from_numpy(y_val_csv).long(), 12).float()

    X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled = scaled_data(X_train_csv, X_test_csv, X_val_csv)

    print('Y_train_csv shape : ', Y_train_csv.shape)
    print('Y_test_csv shape : ', Y_test_csv.shape)
    print('Y_val_csv shape : ', Y_val_csv.shape)

    X_train_1, X_rem_1, y_train_1, y_rem_1 = train_test_split(img_1, label_1,
                                                              train_size=0.6,
                                                              random_state=42,
                                                              )

    X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(X_rem_1, y_rem_1,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            )
    print('*' * 20)
    print('X_train_1 shape : ', X_train_1.shape)
    print('X_test_1 shape : ', X_test_1.shape)
    print('X_val_1 shape : ', X_val_1.shape)
    print('y_train_1 shape : ', y_train_1.shape)
    print('y_test_1 shape : ', y_test_1.shape)
    print('y_val_1 shape : ', y_val_1.shape)

    Y_train_1 = torch.nn.functional.one_hot(torch.from_numpy(y_train_1).long(), 12).float()
    Y_test_1 = torch.nn.functional.one_hot(torch.from_numpy(y_test_1).long(), 12).float()
    Y_val_1 = torch.nn.functional.one_hot(torch.from_numpy(y_val_1).long(), 12).float()

    print('Y_train_1 shape : ', Y_train_1.shape)
    print('Y_test_1 shape : ', Y_test_1.shape)
    print('Y_val_1 shape : ', Y_val_1.shape)

    X_train_2, X_rem_2, y_train_2, y_rem_2 = train_test_split(img_2, label_2,
                                                              train_size=0.6,
                                                              random_state=42,
                                                              )

    X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(X_rem_2, y_rem_2,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            )

    print('*' * 20)
    print('X_train_2 shape : ', X_train_2.shape)
    print('X_test_2 shape : ', X_test_2.shape)
    print('X_val_2 shape : ', X_val_2.shape)
    print('y_train_2 shape : ', y_train_2.shape)
    print('y_test_2 shape : ', y_test_2.shape)
    print('y_val_2 shape : ', y_val_2.shape)

    Y_train_2 = torch.nn.functional.one_hot(torch.from_numpy(y_train_2).long(), 12).float()
    Y_test_2 = torch.nn.functional.one_hot(torch.from_numpy(y_test_2).long(), 12).float()
    Y_val_2 = torch.nn.functional.one_hot(torch.from_numpy(y_val_2).long(), 12).float()

    print('Y_train_2 shape : ', Y_train_2.shape)
    print('Y_test_2 shape : ', Y_test_2.shape)
    print('Y_val_2 shape : ', Y_val_2.shape)


    print((y_train_1 == y_train_csv).all())
    print((y_train_2 == y_train_csv).all())

    print((y_val_1 == y_val_csv).all())
    print((y_val_2 == y_val_csv).all())

    print((y_test_1 == y_test_csv).all())
    print((y_test_2 == y_test_csv).all())


    shape1, shape2 = 32, 32
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], shape1, shape2, 1)
    X_train_2 = X_train_2.reshape(X_train_2.shape[0], shape1, shape2, 1)
    X_val_1 = X_val_1.reshape(X_val_1.shape[0], shape1, shape2, 1)
    X_val_2 = X_val_2.reshape(X_val_2.shape[0], shape1, shape2, 1)
    X_test_1 = X_test_1.reshape(X_test_1.shape[0], shape1, shape2, 1)
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], shape1, shape2, 1)

    X_train_1_scaled = X_train_1 / 255.0
    X_train_2_scaled = X_train_2 / 255.0

    X_val_1_scaled = X_val_1 / 255.0
    X_val_2_scaled = X_val_2 / 255.0

    X_test_1_scaled = X_test_1 / 255.0
    X_test_2_scaled = X_test_2 / 255.0

    print(X_train_1_scaled.shape)
    print(X_test_1_scaled.shape)
    print(X_val_1_scaled.shape)

    print(X_train_2_scaled.shape)
    print(X_test_2_scaled.shape)
    print(X_val_2_scaled.shape)

    return X_train_csv_scaled,X_test_csv_scaled,X_val_csv_scaled,\
        Y_train_csv,Y_test_csv,Y_val_csv,\
        X_train_1_scaled,X_test_1_scaled,X_val_1_scaled,\
        Y_train_1,Y_test_1,Y_val_1,\
        X_train_2_scaled,X_test_2_scaled,X_val_2_scaled,\
        Y_train_2,Y_test_2,Y_val_2

def splitForClients(total_client,ratios,
                    X_train_csv_scaled,X_test_csv_scaled,X_val_csv_scaled,
                    Y_train_csv,Y_test_csv,Y_val_csv,
                    X_train_1_scaled,X_test_1_scaled,X_val_1_scaled,
                    Y_train_1,Y_test_1,Y_val_1,
                    X_train_2_scaled,X_test_2_scaled,X_val_2_scaled,
                    Y_train_2,Y_test_2,Y_val_2):
    # split train data
    # 样本数量
    total_samples = X_train_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_train_csv_scaled_splits = {}
    Y_train_csv_splits = {}
    X_train_1_scaled_splits = {}
    Y_train_1_splits = {}
    X_train_2_scaled_splits = {}
    Y_train_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_train_csv_scaled_splits[clientId] = X_train_csv_scaled[indices[start_index:end_index]]
        Y_train_csv_splits[clientId] = Y_train_csv[indices[start_index:end_index]]
        X_train_1_scaled_splits[clientId] = X_train_1_scaled[indices[start_index:end_index]]
        Y_train_1_splits[clientId] = Y_train_1[indices[start_index:end_index]]
        X_train_2_scaled_splits[clientId] = X_train_2_scaled[indices[start_index:end_index]]
        Y_train_2_splits[clientId] = Y_train_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1

    # split val data=============================================================
    # 样本数量
    ratios = [1 / total_client] * total_client
    total_samples = X_val_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_val_csv_scaled_splits = {}
    Y_val_csv_splits = {}
    X_val_1_scaled_splits = {}
    Y_val_1_splits = {}
    X_val_2_scaled_splits = {}
    Y_val_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_val_csv_scaled_splits[clientId] = X_val_csv_scaled[indices[start_index:end_index]]
        Y_val_csv_splits[clientId] = Y_val_csv[indices[start_index:end_index]]
        X_val_1_scaled_splits[clientId] = X_val_1_scaled[indices[start_index:end_index]]
        Y_val_1_splits[clientId] = Y_val_1[indices[start_index:end_index]]
        X_val_2_scaled_splits[clientId] = X_val_2_scaled[indices[start_index:end_index]]
        Y_val_2_splits[clientId] = Y_val_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1

    # split test data=====================================================
    # 样本数量
    ratios = [1 / total_client] * total_client
    total_samples = X_test_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_test_csv_scaled_splits = {}
    Y_test_csv_splits = {}
    X_test_1_scaled_splits = {}
    Y_test_1_splits = {}
    X_test_2_scaled_splits = {}
    Y_test_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_test_csv_scaled_splits[clientId] = X_test_csv_scaled[indices[start_index:end_index]]
        Y_test_csv_splits[clientId] = Y_test_csv[indices[start_index:end_index]]
        X_test_1_scaled_splits[clientId] = X_test_1_scaled[indices[start_index:end_index]]
        Y_test_1_splits[clientId] = Y_test_1[indices[start_index:end_index]]
        X_test_2_scaled_splits[clientId] = X_test_2_scaled[indices[start_index:end_index]]
        Y_test_2_splits[clientId] = Y_test_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1
    return X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
        Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
        X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
        Y_train_2_splits, Y_test_2_splits, Y_val_2_splits

def calculate_relative_loss_reduction_as_list(client_losses):
    """
    计算每个 client 的局部训练损失相对下降幅度 RF_loss，并以列表形式返回。

    参数：
    - client_losses (dict): 一个字典，key 是 client id，value 是一个 list，表示该 client 各轮训练的 loss。

    返回：
    - rf_losses_list (list): 按输入顺序返回每个 client 的 RF_loss。
    """
    # 计算每个 client 的起始损失和结束损失差值
    loss_reductions = {}
    for client_id, losses in client_losses.items():
        if len(losses) < 2:
            raise ValueError(f"Client {client_id} 的训练损失数据不足，至少需要两轮的损失值。")
        loss_start = losses[0]
        loss_end = losses[-1]
        loss_reductions[client_id] = loss_start - loss_end

    # 找到最大损失下降值
    max_loss_reduction = max(loss_reductions.values())

    if max_loss_reduction == 0:
        raise ValueError("所有 client 的损失下降值均为 0，无法计算相对下降幅度。")

    # 计算相对下降幅度，并以列表形式返回
    rf_losses_list = [
        reduction / max_loss_reduction for reduction in loss_reductions.values()
    ]

    return rf_losses_list

def calculate_relative_train_accuracy(client_acc):
    """
    计算每个 client 的局部训练精度 RF_ACC_Train，并返回一个列表。

    参数：
    - client_acc (dict): 一个字典，key 是 client id，value 是该 client 的训练精度。

    返回：
    - rf_acc_train_list (list): 按输入顺序返回每个 client 的 RF_ACC_Train。
    """
    # 找到最大训练精度
    max_acc = max(client_acc.values())

    if max_acc == 0:
        raise ValueError("所有 client 的训练精度均为 0，无法计算相对训练精度。")

    # 计算相对训练精度，并以列表形式返回
    rf_acc_train_list = [
        acc / max_acc for acc in client_acc.values()
    ]

    return rf_acc_train_list


def calculate_global_validation_accuracy(train_acc, global_acc):
    """
    计算每个 client 的全局验证精度 RF_ACC_Global，并返回一个列表。

    参数：
    - train_acc (dict): 一个字典，key 是 client id，value 是该 client 的训练精度。
    - global_acc (dict): 一个字典，key 是 client id，value 是该 client 的全局验证精度。

    返回：
    - rf_acc_global_list (list): 按输入顺序返回每个 client 的 RF_ACC_Global。
    """
    # 检查两个字典是否对齐
    if set(train_acc.keys()) != set(global_acc.keys()):
        raise ValueError("训练精度和全局验证精度的客户端 ID 不一致。")

    # 计算全局验证精度的最大值
    max_global_acc = max(global_acc.values())
    if max_global_acc == 0:
        raise ValueError("所有 client 的全局验证精度均为 0，无法计算 RF_ACC_Global。")

    # 计算全局验证精度与训练精度的差值及其最大值
    global_train_diff = {client_id: global_acc[client_id] - train_acc[client_id] for client_id in train_acc}
    max_global_train_diff = max(global_train_diff.values())
    if max_global_train_diff == 0:
        raise ValueError("所有 client 的全局验证与训练精度差值均为 0，无法计算 RF_ACC_Global。")

    # 计算 RF_ACC_Global
    rf_acc_global_list = [
        (global_acc[client_id] / max_global_acc) - (global_train_diff[client_id] / max_global_train_diff)
        for client_id in train_acc
    ]

    return rf_acc_global_list

def calculate_loss_outliers(client_losses, lambda_loss=1.5):
    """
    计算每个 client 的损失异常 P_loss，并返回一个列表。

    参数：
    - client_losses (dict): 一个字典，key 是 client id，value 是一个包含 j 轮训练损失的列表。
    - lambda_loss (float): 调节标准差影响的系数，默认值为 1.5。

    返回：
    - loss_outliers (list): 按输入顺序返回每个 client 的损失异常分数 P_loss。
    """
    # 提取每个 client 的最终损失
    final_losses = {client_id: losses[-1] for client_id, losses in client_losses.items()}

    # 计算最终损失的均值和标准差
    loss_values = np.array(list(final_losses.values()))
    # loss_values = np.array([value.cpu().numpy() for value in final_losses.values()])
    # loss_values = np.array([value.detach().cpu().numpy() for value in final_losses.values()])

    # 假设 final_losses 是一个字典，其中的值可能是 GPU Tensor
    # loss_values = []
    # for value in final_losses.values():
    #     if isinstance(value, torch.Tensor):
    #         # 检查是否在 GPU 上，并且需要迁移到 CPU
    #         value = value.detach().cpu().numpy() if value.is_cuda else value.detach().numpy()
    #         loss_values.append(value)
    #     else:
    #         raise TypeError(f"Unexpected type {type(value)} in final_losses.values()")

    loss_values = np.array(loss_values)

    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)

    # 计算阈值
    threshold = mean_loss + lambda_loss * std_loss

    # 计算最终损失的最大值
    max_loss = np.max(loss_values)

    # 避免除以零的情况
    if max_loss == 0:
        raise ValueError("所有 client 的最终损失均为 0，无法计算损失异常分数。")

    # 计算损失异常分数
    loss_outliers = [
        final_loss / max_loss if final_loss > threshold else 0
        for final_loss in loss_values
    ]

    return loss_outliers


def calculate_performance_bias(val_acc, global_acc):
    """
    计算每个 client 的性能偏离 P_bias，并返回一个列表。

    参数：
    - val_acc (dict): 一个字典，key 是 client id，value 是该 client 的验证精度。
    - global_acc (dict): 一个字典，key 是 client id，value 是该 client 的全局验证精度。

    返回：
    - performance_bias_list (list): 按输入顺序返回每个 client 的性能偏离值 P_bias。
    """
    # 检查两个字典是否对齐
    if set(val_acc.keys()) != set(global_acc.keys()):
        raise ValueError("验证精度和全局验证精度的客户端 ID 不一致。")

    # 计算性能偏离
    performance_bias_list = []
    for client_id in val_acc:
        val = val_acc[client_id]
        global_val = global_acc[client_id]
        max_val = max(val, global_val)

        # 避免除以零的情况
        if max_val == 0:
            performance_bias = 0  # 如果验证和全局验证精度均为 0，则偏离值为 0
        else:
            performance_bias = abs(val - global_val) / max_val

        performance_bias_list.append(performance_bias)

    return performance_bias_list

def pareto_optimization(
    rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias, client_num
):
    """
    实现 Pareto 优化，筛选节点。

    参数：
    - rf_loss (list): 局部训练损失相对下降幅度。
    - rf_acc_train (list): 局部训练精度。
    - rf_acc_val (list): 局部验证精度。
    - rf_acc_global (list): 全局验证精度。
    - p_loss (list): 损失异常。
    - p_bias (list): 性能偏离。
    - client_num (int): 要选出的节点数。

    返回：
    - selected_clients (list): 选中的 client ID（按输入顺序从 0 开始）。
    """
    # 将输入指标整合为二维数组，便于处理
    # data = np.array([rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, -np.array(p_loss), -np.array(p_bias)]).T

    # 确保所有数组中的元素都转换为 NumPy 数组
    # rf_loss = np.array([x.detach().cpu().numpy() for x in rf_loss])
    rf_loss = np.array(list(rf_loss))
    rf_acc_train = rf_acc_train.detach().cpu().numpy() if isinstance(rf_acc_train, torch.Tensor) else np.array(
        rf_acc_train)
    rf_acc_val = rf_acc_val.detach().cpu().numpy() if isinstance(rf_acc_val, torch.Tensor) else np.array(rf_acc_val)
    rf_acc_global = rf_acc_global.detach().cpu().numpy() if isinstance(rf_acc_global, torch.Tensor) else np.array(
        rf_acc_global)
    p_loss = p_loss.detach().cpu().numpy() if isinstance(p_loss, torch.Tensor) else np.array(p_loss)
    p_bias = p_bias.detach().cpu().numpy() if isinstance(p_bias, torch.Tensor) else np.array(p_bias)
    # rf_acc_train = np.array([x.detach().cpu().numpy() for x in rf_acc_train])
    # rf_acc_val = np.array([x.detach().cpu().numpy() for x in rf_acc_val])
    # rf_acc_global = np.array([x.detach().cpu().numpy() for x in rf_acc_global])
    # p_loss = np.array([x.detach().cpu().numpy() for x in p_loss])
    # p_bias = np.array([x.detach().cpu().numpy() for x in p_bias])

    # 构造 NumPy 数组并转置
    data = np.array([rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, -p_loss, -p_bias]).T

    # Pareto 前沿筛选
    def is_dominated(point, others):
        """判断 point 是否被 others 支配"""
        return any(np.all(other >= point) and np.any(other > point) for other in others)

    pareto_indices = [
        i for i, point in enumerate(data) if not is_dominated(point, np.delete(data, i, axis=0))
    ]
    pareto_clients = pareto_indices

    # 如果前沿节点数多于 client_num，随机选取
    if len(pareto_clients) > client_num:
        return random.sample(pareto_clients, client_num)

    # 如果前沿节点数小于 client_num，基于组合评分补充
    remaining_slots = client_num - len(pareto_clients)
    pareto_scores = [0.4 * rf_loss[i] + 0.6 * rf_acc_global[i] for i in range(len(rf_loss))]
    sorted_indices = np.argsort(pareto_scores)[::-1]  # 按评分从高到低排序

    selected_clients = set(pareto_clients)
    for i in sorted_indices:
        if len(selected_clients) >= client_num:
            break
        if i not in selected_clients:
            selected_clients.add(i)

    return list(selected_clients)

def get_top_clients_with5RF(rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias, client_num):
    # rf_loss = np.array([x.detach().cpu().numpy() for x in rf_loss])
    rf_loss = np.array(list(rf_loss))
    rf_acc_train = rf_acc_train.detach().cpu().numpy() if isinstance(rf_acc_train, torch.Tensor) else np.array(
        rf_acc_train)
    rf_acc_val = rf_acc_val.detach().cpu().numpy() if isinstance(rf_acc_val, torch.Tensor) else np.array(rf_acc_val)
    rf_acc_global = rf_acc_global.detach().cpu().numpy() if isinstance(rf_acc_global, torch.Tensor) else np.array(
        rf_acc_global)
    p_loss = p_loss.detach().cpu().numpy() if isinstance(p_loss, torch.Tensor) else np.array(p_loss)
    p_bias = p_bias.detach().cpu().numpy() if isinstance(p_bias, torch.Tensor) else np.array(p_bias)

    # 计算综合评分
    scores = (
            0.2 * rf_loss +
            0.1 * rf_acc_train +
            0.2 * rf_acc_val +
            0.3 * rf_acc_global -
            0.1 * p_loss -
            0.1 * p_bias
    )
    origin_scores = scores
    # 获取得分最高的前 clientNum 个客户端 ID
    top_client_ids = np.argsort(scores)[::-1][:client_num]  # 降序排序，取前 clientNum 个
    return top_client_ids.tolist(),origin_scores

class ModelCSVIMG(nn.Module):
    # def __init__(self, num_csv_features, img_shape1, img_shape2):
    #     super(ModelCSVIMG, self).__init__()
    #
    #     # 第一输入分支：处理CSV特征的1D卷积
    #     # self.conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3)
    #     # self.pool1d = nn.MaxPool1d(kernel_size=2)
    #     # self.batch_norm1d = nn.BatchNorm1d(10)
    #     self.fc_csv_1 = nn.Linear(num_csv_features, 2000)
    #     self.bn_csv_1 = nn.BatchNorm1d(2000)
    #     self.fc_csv_2 = nn.Linear(2000, 600)
    #     self.bn_csv_2 = nn.BatchNorm1d(600)
    #     self.dropout_csv = nn.Dropout(0.2)
    #     # self.fc_csv_3 = nn.Linear(600, 12)
    #
    #     # 第二输入分支：处理第一张图像的2D卷积
    #     self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    #     self.pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.batch_norm2d_1 = nn.BatchNorm2d(16)
    #
    #     # 第三输入分支：处理第二张图像的2D卷积
    #     self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    #     self.pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
    #     self.batch_norm2d_2 = nn.BatchNorm2d(16)
    #
    #     # 计算展平后的维度
    #     # self.flattened_dim_csv = (num_csv_features - 2) // 2 * 10
    #     self.flattened_dim_csv = 600
    #     self.flattened_dim_img = 16 * (img_shape1 // 2) * (img_shape2 // 2)
    #
    #     # 全连接层
    #     self.fc1 = nn.Linear(self.flattened_dim_csv + 2 * self.flattened_dim_img, 600)
    #     self.fc2 = nn.Linear(600, 1200)
    #     self.dropout = nn.Dropout(0.2)
    #     self.fc3 = nn.Linear(1200, 12)
    #
    # def forward(self, x_csv, x_img1, x_img2):
    #     # 第一分支：CSV特征
    #     # x_csv = F.relu(self.conv1d(x_csv))
    #     # x_csv = self.pool1d(x_csv)
    #     # x_csv = self.batch_norm1d(x_csv)
    #     # x_csv = x_csv.view(x_csv.size(0), -1)  # 展平
    #
    #     x_csv = F.relu(self.bn_csv_1(self.fc_csv_1(x_csv)))
    #     x_csv = F.relu(self.bn_csv_2(self.fc_csv_2(x_csv)))
    #     x_csv = self.dropout_csv(x_csv)
    #     # x_csv = self.fc_csv_3(x_csv)
    #
    #     x_img1 = x_img1.permute(0, 3, 1, 2)
    #     # 第二分支：第一张图像
    #     x_img1 = F.relu(self.conv2d_1(x_img1))
    #     x_img1 = self.pool2d_1(x_img1)
    #     x_img1 = self.batch_norm2d_1(x_img1)
    #     x_img1 = x_img1.view(x_img1.size(0), -1)  # 展平
    #
    #     x_img2 = x_img2.permute(0, 3, 1, 2)
    #     # 第三分支：第二张图像
    #     x_img2 = F.relu(self.conv2d_2(x_img2))
    #     x_img2 = self.pool2d_2(x_img2)
    #     x_img2 = self.batch_norm2d_2(x_img2)
    #     x_img2 = x_img2.view(x_img2.size(0), -1)  # 展平
    #
    #     # 连接三个分支
    #     x = torch.cat((x_csv, x_img1, x_img2), dim=1)
    #
    #     # 全连接层
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.dropout(x)
    #     x = F.softmax(self.fc3(x), dim=1)
    #
    #     return x
    def __init__(self, num_csv_features, img_shape1, img_shape2):
        super(ModelCSVIMG, self).__init__()

        # V1======================================================================================
        # # 第一输入分支：处理CSV特征的1D卷积
        # # self.conv1d = nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3)
        # # self.pool1d = nn.MaxPool1d(kernel_size=2)
        # # self.batch_norm1d = nn.BatchNorm1d(10)
        # self.fc_csv_1 = nn.Linear(num_csv_features, 2000)
        # self.bn_csv_1 = nn.BatchNorm1d(2000)
        # self.fc_csv_2 = nn.Linear(2000, 600)
        # self.bn_csv_2 = nn.BatchNorm1d(600)
        # self.dropout_csv = nn.Dropout(0.2)
        # # self.fc_csv_3 = nn.Linear(600, 12)
        #
        # # 第二输入分支：处理第一张图像的2D卷积
        # self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.pool2d_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batch_norm2d_1 = nn.BatchNorm2d(16)
        #
        # # 第三输入分支：处理第二张图像的2D卷积
        # self.conv2d_2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # self.pool2d_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.batch_norm2d_2 = nn.BatchNorm2d(16)
        #
        # # 计算展平后的维度
        # # self.flattened_dim_csv = (num_csv_features - 2) // 2 * 10
        # self.flattened_dim_csv = 600
        # self.flattened_dim_img = 16 * (img_shape1 // 2) * (img_shape2 // 2)
        #
        # # 全连接层
        # self.fc1 = nn.Linear(self.flattened_dim_csv + 2 * self.flattened_dim_img, 600)
        # self.fc2 = nn.Linear(600, 1200)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1200, 12)

        # # v2==========================================
        # # 第一输入分支：处理CSV特征
        # self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        # self.csv_bn_1 = nn.BatchNorm1d(2000)
        # self.csv_fc_2 = nn.Linear(2000, 600)
        # self.csv_bn_2 = nn.BatchNorm1d(600)
        # self.csv_fc_3 = nn.Linear(600, 100)
        # self.csv_dropout = nn.Dropout(0.2)
        #
        # # 第二输入分支：处理第一张图像的2D卷积
        # self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        # self.img1_batch_norm = nn.BatchNorm2d(18)
        # self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        # self.img1_dropout = nn.Dropout(0.2)
        #
        # # 第三输入分支：处理第二张图像的2D卷积
        # self.img2_conv = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        # self.img2_batch_norm = nn.BatchNorm2d(18)
        # self.img2_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.img2_fc1 = nn.Linear(18 * (16) * 16, 100)
        # self.img2_dropout = nn.Dropout(0.2)
        #
        # # 全连接层
        # self.fc1 = nn.Linear(300, 600)
        # self.fc2 = nn.Linear(600, 1200)
        # self.dropout = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(1200, 12)

        # v3==========================================
        # 第一输入分支：处理CSV特征
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
        self.fc2 = nn.Linear(2000, 12)

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
        # x += residual
        x = torch.cat((residual, x), dim=1)
        x = F.softmax(self.fc2(x), dim=1)

        return x


def trainValModelCSVIMG(model_name, svmethod, total_client,num_clients,epoch,max_acc,epoch_size,local_epoch_per_round,round_early_stop,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits,
                        X_train_1_scaled_splits, X_test_1_scaled_splits,
                        X_train_2_scaled_splits, X_test_2_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits):
    # Instantiate the model    the total_client’th split used for server
    # input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    model_MLP = ModelCSVIMG(X_train_csv_scaled_splits[0].shape[1],32,32)
    # model_MLP = model_MLP.double()
    model_MLP = model_MLP.to(device)

    # initialize server and clients
    server = Server(model_MLP,epoch_size, [X_test_csv_scaled_splits[total_client-1],X_test_1_scaled_splits[total_client-1],X_test_2_scaled_splits[total_client-1],Y_test_csv_splits[total_client-1]], num_clients)
    clients = []
    for client_index in range(total_client):
        clients.append(Client(epoch_size = epoch_size, local_epoch_per_round = local_epoch_per_round,
                              train_dataset = [X_train_csv_scaled_splits[client_index], X_train_1_scaled_splits[client_index], X_train_2_scaled_splits[client_index],
                               Y_train_csv_splits[client_index]],
                              val_dataset =  [X_test_csv_scaled_splits[client_index], X_test_1_scaled_splits[client_index], X_test_2_scaled_splits[client_index],
                               Y_test_csv_splits[client_index]], id = client_index))
        # clients.append(Client(model_MLP, epoch_size, local_epoch_per_round,
        #                       [X_train_csv_scaled_splits[c], X_train_1_scaled_splits[c], X_train_2_scaled_splits[c],
        #                        Y_train_csv_splits[c]],
        #                       [X_test_csv_scaled_splits[c], X_test_1_scaled_splits[c], X_test_2_scaled_splits[c],
        #                        Y_test_csv_splits[c]], id=c))

    # train
    clients_scoresDict = {}

    clients_scoresDict_top = {}

    perEpoch_clients_losses = {}
    perEpoch_clients_train_acc = {}
    perEpoch_clients_local_test_acc = {}
    perEpoch_clients_global_test_acc = {}

    clients_train_acc = {}
    clients_train_loss = {}
    clients_test_acc = {}
    clients_test_loss = {}
    clients_rf_relative_loss_reduction = {}
    clients_rf_acc_train = {}
    clients_rf_global_validation_accuracy ={}
    clients_rf_loss_outliers = {}
    clients_rf_performance_bias = {}
    clients_epoch_selected = {}
    for i in range(total_client + 1):  # one more for server
        epoch_train_acc = []
        epoch_train_loss = []
        epoch_test_acc = []
        epoch_test_loss = []
        epoch_selected = []
        rf_relative_loss_reduction = []
        rf_acc_train = []
        rf_acc_test = []
        rf_global_validation_accuracy = []
        rf_loss_outliers = []
        rf_performance_bias = []
        one_client_score = []
        clients_train_acc[i] = epoch_train_acc
        clients_train_loss[i] = epoch_train_loss
        clients_test_acc[i] = epoch_test_acc
        clients_test_loss[i] = epoch_test_loss
        clients_scoresDict[i] = one_client_score
        clients_rf_relative_loss_reduction[i] = rf_relative_loss_reduction
        clients_rf_acc_train[i] = rf_acc_train
        clients_rf_global_validation_accuracy[i] = rf_global_validation_accuracy
        clients_rf_loss_outliers[i] = rf_loss_outliers
        clients_rf_performance_bias[i] = rf_performance_bias
        clients_epoch_selected[i] = epoch_selected
    epoch_count = 0
    for e in range(epoch):
        epoch_count += 1
        print('round_{}'.format(e))
        if epoch_count < round_early_stop+1:
            diff_client = {}
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name] = torch.zeros_like(params)  # initialize weight_accumulator
            for client_index in range(total_client):
                round_client_model, diff, test_acc_client, loss_client, min_loss, max_loss, losses, train_acc= clients[client_index].local_train(server.global_model)
                print('clent_{}, trainAcc:{}, trainLoss:{}, testAcc:{}'.format(client_index,train_acc,loss_client,test_acc_client))
                perEpoch_clients_losses[client_index] = losses
                perEpoch_clients_train_acc[client_index] = train_acc
                perEpoch_clients_local_test_acc[client_index] = test_acc_client

                # value client model by test data
                total_loss = 0.0
                correct = 0
                dataset_size = 0
                for test_data_index in range(total_client):
                    test_server_loader = torch.utils.data.DataLoader(
                        CostumDataset(X_test_csv_scaled_splits[test_data_index], X_test_1_scaled_splits[test_data_index],
                                      X_test_2_scaled_splits[test_data_index], Y_test_csv_splits[test_data_index]),
                        batch_size=epoch_size)
                    round_client_model.eval()
                    for batch_id, batch in enumerate(test_server_loader):
                        data1 = batch[0]
                        data2 = batch[1]
                        data3 = batch[2]
                        target = torch.squeeze(batch[3])
                        # target = torch.squeeze(batch[3]).float()

                        dataset_size += data1.size()[0]
                        data1 = data1.to(device).float()
                        data2 = data2.to(device).float()
                        data3 = data3.to(device).float()
                        target = target.to(device).float()

                        output = round_client_model(data1, data2, data3)
                        total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

                        pred = output.detach().max(1)[1]
                        correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

                oneclient_test_acc = 100.0 * (float(correct) / float(dataset_size))
                oneclient_test_loss = total_loss / dataset_size

                client_score = evaluate_model(test_acc_client, loss_client, min_loss, max_loss, oneclient_test_acc,
                                              oneclient_test_loss)
                clients_scoresDict_top[client_index] = client_score

                perEpoch_clients_global_test_acc[client_index]=oneclient_test_acc

                clients_train_acc[client_index].append(test_acc_client)
                clients_train_loss[client_index].append(loss_client)
                clients_test_acc[client_index].append(oneclient_test_acc)
                clients_test_loss[client_index].append(oneclient_test_loss)
                clients_epoch_selected[client_index].append(0)
                diff_client[client_index] = diff

            rf_relative_loss_reduction = calculate_relative_loss_reduction_as_list(perEpoch_clients_losses)
            rf_acc_train = calculate_relative_train_accuracy(perEpoch_clients_train_acc)
            rf_global_validation_accuracy =calculate_global_validation_accuracy(perEpoch_clients_train_acc,perEpoch_clients_global_test_acc)
            rf_loss_outliers = calculate_loss_outliers(perEpoch_clients_losses)
            rf_performance_bias = calculate_performance_bias(perEpoch_clients_local_test_acc,perEpoch_clients_global_test_acc)

            for client_index in range(total_client):
                clients_rf_relative_loss_reduction[client_index].append(rf_relative_loss_reduction[client_index])
                clients_rf_acc_train[client_index].append(rf_acc_train[client_index])
                clients_rf_global_validation_accuracy[client_index].append(rf_global_validation_accuracy[client_index])
                clients_rf_loss_outliers[client_index].append(rf_loss_outliers[client_index])
                clients_rf_performance_bias[client_index].append(rf_performance_bias[client_index])

            if svmethod == '5RF':
                candidates,scores = get_top_clients_with5RF(rf_relative_loss_reduction, rf_acc_train, rf_acc_test,
                                                 rf_global_validation_accuracy, rf_loss_outliers, rf_performance_bias,
                                                 num_clients)
                for index in range(len(scores)):
                    clients_scoresDict[index].append(scores[index])
            elif svmethod == '4RF':
                candidates = get_top_clients(clients_scoresDict_top, num_clients)
                for key in clients_scoresDict_top.keys():
                    clients_scoresDict[key].append(clients_scoresDict_top[key])
            elif svmethod =='pareto':
                candidates = pareto_optimization(rf_relative_loss_reduction, rf_acc_train, rf_acc_test,
                                                 rf_global_validation_accuracy, rf_loss_outliers, rf_performance_bias,
                                                 num_clients)
            elif svmethod =='random':
                candidates = np.random.choice(total_client, num_clients, replace=False)

            for selected_client_index in candidates:
                clients_epoch_selected[selected_client_index][-1] = 1
            for slected_client_index in candidates:
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff_client[slected_client_index][name])  # update weight_accumulator
            server.model_aggregate(weight_accumulator)  # aggregate global model
            # acc, loss = server.model_eval()
            y_test, y_predict = [], []
            total_loss = 0.0
            correct = 0
            dataset_size = 0
            for test_data_index in range(total_client):
                test_server_loader = torch.utils.data.DataLoader(
                    CostumDataset(X_test_csv_scaled_splits[test_data_index], X_test_1_scaled_splits[test_data_index],
                                  X_test_2_scaled_splits[test_data_index], Y_test_csv_splits[test_data_index]),
                    batch_size=epoch_size)
                server.global_model.eval()
                for batch_id, batch in enumerate(test_server_loader):
                    data1 = batch[0]
                    data2 = batch[1]
                    data3 = batch[2]
                    target = torch.squeeze(batch[3])
                    # target = torch.tensor(target, dtype=torch.int64)
                    # target = torch.squeeze(batch[3]).float()

                    dataset_size += data1.size()[0]
                    data1 = data1.to(device).float()
                    data2 = data2.to(device).float()
                    data3 = data3.to(device).float()
                    target = target.to(device).float()

                    output = server.global_model(data1, data2, data3)
                    total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

                    pred = output.detach().max(1)[1]
                    y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
                    y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())
                    correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(dataset_size))
            loss = total_loss / dataset_size
            clients_train_acc[total_client].append(acc)
            clients_train_loss[total_client].append(loss)
            clients_test_acc[total_client].append(acc)
            clients_test_loss[total_client].append(loss)
            clients_epoch_selected[total_client].append(0)
            print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))
            if acc > max_acc:
                max_acc = acc
                torch.save(server.global_model.state_dict(),
                       "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}_epoch_size_{}_local_epoch_{}_svmethod_{}.pth".format(model_name, total_client,
                                                                                            num_clients, epoch,
                                                                                            epoch_size,local_epoch_per_round,svmethod))
                print("save model")
                epoch_count = 0
        else:
            break
        # test
    model = model_MLP
    model.load_state_dict(torch.load(
        "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}_epoch_size_{}_local_epoch_{}_svmethod_{}.pth".format(model_name, total_client, num_clients,
                                                                             epoch,
                                                                             epoch_size,local_epoch_per_round,svmethod)))
    model = model.to(device)
    model.eval()
    y_test, y_predict = [], []

    for test_data_index in range(total_client):
        test_server_loader = torch.utils.data.DataLoader(
            CostumDataset(X_test_csv_scaled_splits[test_data_index], X_test_1_scaled_splits[test_data_index],
                          X_test_2_scaled_splits[test_data_index], Y_test_csv_splits[test_data_index]),
            batch_size=epoch_size)
        for batch_id, batch in enumerate(test_server_loader):
            data1 = batch[0]
            data2 = batch[1]
            data3 = batch[2]
            target = torch.squeeze(batch[3])
            # target = torch.tensor(target, dtype=torch.int64)
            # target = torch.squeeze(batch[3]).float()

            dataset_size += data1.size()[0]
            data1 = data1.to(device).float()
            data2 = data2.to(device).float()
            data3 = data3.to(device).float()
            target = target.to(device).float()

            output = model(data1, data2, data3)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
            y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    loss = total_loss / dataset_size

    print('server_test_acc', acc)
    print('server_test_loss', loss)

    # classification_report
    # print(classification_report(y_test, y_predict,
    #                             target_names=[f'A{i}' for i in range(1, 13)], digits=4))

    print('max_acc', max_acc)
    csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}_epochSize_{}_local_epoch_{}_svmethod_{}.csv".format(model_name, total_client,
                                                                                         num_clients, epoch,
                                                                                         epoch_size,local_epoch_per_round,svmethod)
    # 保存到 CSV 文件
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['server_client_index',server_client_index])
        writer.writerow(
            ['client', 'Epoch', 'train_Loss', 'train_Accuracy', 'test_Loss', 'test_Accuracy','rf_loss', 'rf_acc_train', 'rf_acc_val', 'rf_acc_global', 'p_loss', 'p_bias', 'selected'])
        for i in range(total_client+1):
            print('i:',i)
            # 添加列名
            train_losses = clients_train_loss[i]
            train_accs = clients_train_acc[i]
            test_loss = clients_test_loss[i]
            test_acc = clients_test_acc[i]
            rf_loss = clients_rf_relative_loss_reduction[i]
            rf_acc_train = clients_rf_acc_train[i]
            rf_acc_global = clients_rf_global_validation_accuracy[i]
            p_loss = clients_rf_loss_outliers[i]
            p_bias = clients_rf_performance_bias[i]
            selecteds = clients_epoch_selected[i]
            if i == total_client:
                for j in range(len(train_losses)):
                    writer.writerow([i, j + 1, train_losses[j], train_accs[j], test_loss[j], test_acc[j], 0,0,0,0,0,0,0])
            else:
                for j in range(len(train_losses)):
                    writer.writerow([i, j + 1, train_losses[j], train_accs[j], test_loss[j], test_acc[j],rf_loss[j],rf_acc_train[j], rf_acc_global[j],p_loss[j],p_bias[j],selecteds[j]])

    # confusion matrix
    # plt.figure(dpi=150, figsize=(6, 4))
    # classes = [f'A{i}' for i in range(1, 13)]
    # mat = confusion_matrix(y_test, y_predict)

    # df_cm = pd.DataFrame(mat, index=classes, columns=classes)
    # sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    # plt.ylabel('True label', fontsize=15)
    # plt.xlabel('Predicted label', fontsize=15)
    # # 保存图像
    # plt.savefig(
    #     "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(model_name, total_client, num_clients,
    #                                                                          epoch,
    #                                                                          datetime.now().strftime(
    #                                                                              '%Y-%m-%d-%H-%M-%S')))
    # plt.show()


# Server
class Server(object):
    def __init__(self, model, epoch_size, eval_dataset, num_clients):
        
        self.global_model = model
        self.epoch_size = epoch_size
        self.num_clients = num_clients
        self.serverTestDataSet = CostumDataset(eval_dataset[0],eval_dataset[1],eval_dataset[2],eval_dataset[3])
        self.eval_loader = torch.utils.data.DataLoader(self.serverTestDataSet, batch_size=epoch_size)
	
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1/self.num_clients)   # average
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data1 = batch[0]
            data2 = batch[1]
            data3 = batch[2]
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[3])
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            data1 = data1.to(device).float()
            data2 = data2.to(device).float()
            data3 = data3.to(device).float()
            target = target.to(device).float()
            
            output = self.global_model(data1,data2,data3)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 *(float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        return acc, loss


# Client
class Client(object):
    def __init__(self, epoch_size, local_epoch_per_round, train_dataset,val_dataset, id = -1):
                self.epoch_size = epoch_size
                self.local_epoch_per_round = local_epoch_per_round
                self.client_id = id
                self.train_dataset = CostumDataset(train_dataset[0],train_dataset[1],train_dataset[2],train_dataset[3])
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=epoch_size,shuffle=True)
                self.eval_dataset = CostumDataset(val_dataset[0], val_dataset[1], val_dataset[2], val_dataset[3])
                self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=epoch_size,shuffle=False)

    def local_train(self, global_model):
        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        # # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
        # self.local_model.train()
        # min_loss = -100000.00
        # max_loss = 100000.00
        # losses = []
        # accs = []
        # correct = 0
        # dataset_size = 0
        # for e in range(self.local_epoch_per_round):
        #     for batch_id, batch in enumerate(self.train_loader):
        #         data1 = batch[0]
        #         data2 = batch[1]
        #         data3 = batch[2]
        #         target = torch.squeeze(batch[3]).int()
        #         target = torch.tensor(target, dtype=torch.int64)
        #         # target = torch.squeeze(batch[3]).float()
        #         # target = torch.tensor(target, dtype=float)
        #         dataset_size += data1.size()[0]
        #         if torch.cuda.is_available():
        #             data1 = data1.cuda().float()
        #             data2 = data2.cuda().float()
        #             data3 = data3.cuda().float()
        #             target = target.cuda().float()
        #
        #         output = self.local_model(data1,data2,data3)
        #         loss = nn.functional.cross_entropy(output, target)
        #         pred = output.max(1)[1]
        #         correct += pred.eq(target.max(1)[1].view_as(pred)).sum().item()
        #
        #         if loss > max_loss:
        #             max_loss = loss
        #         if loss < min_loss:
        #             min_loss = loss
        #         losses.append(loss)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        # train_acc = 100.0 * (float(correct) / float(dataset_size))
        # # after epoch train, eval model and save client model acc,loss to csv
        # # self.local_model.eval()
        # # total_loss = 0.0
        # # correct = 0
        # # dataset_size = 0
        # # for batch_id, batch in enumerate(self.eval_loader):
        # #     data1 = batch[0]
        # #     data2 = batch[1]
        # #     data3 = batch[2]
        # #     # target = torch.squeeze(batch[1]).int()
        # #     # target = torch.tensor(target, dtype=torch.int64)
        # #     target = torch.squeeze(batch[3]).float()
        # #     # target = torch.tensor(target,dtype=float)
        # #
        # #     dataset_size += data1.size()[0]
        # #
        # #     if torch.cuda.is_available():
        # #         data1 = data1.cuda().double()
        # #         data2 = data2.cuda().double()
        # #         data3 = data3.cuda().double()
        # #         target = target.cuda()
        # #
        # #     output = self.local_model(data1,data2,data3)
        # #     total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
        # #
        # #     pred = output.detach().max(1)[1]
        # #     correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()
        # #
        # # test_acc = 100.0 * (float(correct) / float(dataset_size))
        # # loss = total_loss / dataset_size
        #
        # self.local_model.eval()
        # running_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for batch_id, batch in enumerate(self.eval_loader):
        #         data1 = batch[0].cuda().float()
        #         data2 = batch[1].cuda().float()
        #         data3 = batch[2].cuda().float()
        #         target = torch.squeeze(batch[3]).int()
        #         target = torch.tensor(target, dtype=torch.int64).cuda().float()
        #
        #         # target = torch.squeeze(batch[3]).cuda().float()
        #
        #         output = model(data1, data2, data3)
        #         loss = nn.functional.cross_entropy(output, target)
        #
        #         # 统计
        #         running_loss += loss.item() * data1.size()[0]
        #         _, predicted = output.max(1)
        #         total += target.size(0)
        #         correct += predicted.eq(target.max(1)[1]).sum().item()
        # test_acc = 100.0 * correct / total

        print(self.train_dataset.features1.shape)
        model = ModelCSVIMG(self.train_dataset.features1.shape[1], 32, 32)
        model = model.to(device)

        for name, param in global_model.state_dict().items():
            model.state_dict()[name].copy_(param.clone())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = local_epoch_per_round
        best_acc = 0.0

        min_loss = -100000.00
        max_loss = 100000.00
        losses = []
        for epoch in range(num_epochs):

            # 训练
            train_loss, train_acc = train_one_epoch(model, self.train_loader, criterion, optimizer)
            if train_loss > max_loss:
                max_loss = train_loss
            if train_loss < min_loss:
                min_loss = train_loss
            losses.append(train_loss)

            # 验证
            val_loss, val_acc = validate(model, self.eval_loader, criterion)

            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")


            # 打印每个epoch的结果
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - ")

        print("client_{} 训练完成，最佳验证准确率: {:.2f}%".format(self.client_id, best_acc))

        diff = dict()
        for name, data in model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return model, diff,val_acc,val_loss,min_loss,max_loss,losses,train_acc

def train_one_epoch(model, train_loader, criterion, optimizer):
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


def validate(model, val_loader, criterion):
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



def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def evaluate_model(acc, loss, min_loss, max_loss, oneclient_test_acc, oneclient_test_loss,alpha=0.8, beta=0.8):
    normalized_loss = normalize(loss, min_loss, max_loss)
    # 训练集
    train_score = alpha * acc + (1 - alpha) * normalized_loss

    # 验证集评分
    val_score = beta * oneclient_test_acc + (1 - beta) * oneclient_test_loss

    # 综合评分（可以根据需求进一步调整权重）
    combined_score = (train_score + val_score) / 2
    return combined_score

def get_top_clients(client_dict, num):
    # 根据value（clientScore）排序，返回前num个key
    sorted_clients = sorted(client_dict.items(), key=lambda item: item[1], reverse=True)
    top_clients = [client[0] for client in sorted_clients[:num]]
    return top_clients


def select_nodes_with_dynamic_threshold(node_scores, max_nodes, std_multiplier=1.0):
    """
    动态阈值节点选择算法。

    Args:
        node_scores (dict): 包含节点 ID 和分数的字典，格式为 {node_id: score, ...}。
        max_nodes (int): 可选择的最大节点数量。
        std_multiplier (float): 标准差调整系数，默认为 1.0。

    Returns:
        selected_nodes (list): 被选择的节点 ID 列表。
    """
    # 提取分数
    scores = np.array(list(node_scores.values()))
    node_ids = list(node_scores.keys())

    # 计算平均值和标准差
    mean_score = np.mean(scores)
    std_dev = np.std(scores)

    # 动态阈值
    dynamic_threshold = mean_score + std_multiplier * std_dev

    # 根据动态阈值选择节点
    selected_nodes = [
        node_id for node_id, score in node_scores.items() if score >= dynamic_threshold
    ]

    # 如果选择的节点数量超过限制，按分数从高到低截取前 max_nodes 个
    if len(selected_nodes) > max_nodes:
        selected_nodes = sorted(
            selected_nodes, key=lambda node_id: node_scores[node_id], reverse=True
        )[:max_nodes]

    # 如果未达到数量限制，按分数从高到低补足到 max_nodes
    if len(selected_nodes) < max_nodes:
        remaining_nodes = [
            node_id for node_id in node_scores if node_id not in selected_nodes
        ]
        remaining_nodes = sorted(
            remaining_nodes, key=lambda node_id: node_scores[node_id], reverse=True
        )
        selected_nodes += remaining_nodes[: max_nodes - len(selected_nodes)]

    return selected_nodes

# ModelLost1
# Node csv data replaced
#     for index in [2,3,4,5,6,7,8,9,10,11]:
# model_name = 'tc1c2ResModelV3WithModelLost1'
# model_name = 'tc1c2ResModelV3SCV3WithModelLost1'
# 模态丢失、数据不平衡
# ratios = [4 / 100, 4 / 100, 4 / 100, 4 / 100, 1 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200]
# 2,3,4,5,6,7,8,9,10,11] no csv
# model_name = 'tc1c2ResModelV3SCV3WithBiasModelBiasRation1'
# model_name = 'tc1c2ResModelV3WithBiasModelBiasRation1'

# ModelLost1
# Node csv data replaced
#     for index in [2,3,4,5,6,7,8,9,10,11]:
# noModelLostWithNodeBias
# model_name = 'tc1c2ResModelV3DataV3Adam'

#ModelLost
# model_name = 'tc1c2ResModelV3DataV3AdamWithSCVLost'
model_names = {'tc1c2ResModelV3DataV3Adam','tc1c2ResModelV3DataV3AdamWithSCVLost','tc1c2ResModelV3DataV3AdamWithImgLost'}

# set_seed()
# hyperparameters
max_acc = 30  # thorshold of accuracy (80%), for saving best model
epoch = 200
epoch_size = 64
total_client = 12  # total number of clients
num_clients = 6  # number of clients selected per round
local_epoch_per_round = 3
round_early_stop = 10
svmethod ='pareto'
# svmethod = '5RF'
# svmethod = '4RF'
# svmethod ='random'
svmethods = {'pareto','5RF','random'}

server_client_index = random.randint(0, total_client-1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    # 定义比例，默认均分
    ratios = [1 / total_client] * total_client
    # ratios = [0.0999,0.0999,0.0999,0.0999,0.0999,0.0777,0.0777,0.0777,0.0777,0.0777,0.0222,0.0222,0.0222,0.0222,0.0223]
    # ratios = [1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,1/20,2/15,2/15,2/15]
    # rations 2
    # ratios = [1/10,1/10,1/10,1/10,1/10, 7/90,7/90,7/90,7/90,7/90,1 / 45, 1 / 45, 1 / 45, 1 / 45, 1 / 45]
    # rations 3
    # ratios = [3 / 10, 3 / 10, 3 / 10, 1 / 120, 1 / 120, 1 / 120,1 / 120,1 / 120,1 / 120,1 / 120,1 / 120,1 / 120,1 / 120,1 / 120,1 / 120]
    # rations 5
    # ratios = [4 / 100, 4 / 100, 4 / 100, 4 / 100, 1 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200, 2 / 200]

    # load data
    # load data
    X_train_csv_scaled_splits, X_test_csv_scaled_splits, \
        Y_train_csv_splits, Y_test_csv_splits, \
        X_train_1_scaled_splits, X_test_1_scaled_splits, \
        Y_train_1_splits, Y_test_1_splits, \
        X_train_2_scaled_splits, X_test_2_scaled_splits, \
        Y_train_2_splits, Y_test_2_splits = loadClientsData()


    # if server_client_index != total_client-1:
    #     X_train_csv_scaled_splits[server_client_index], X_train_csv_scaled_splits[total_client - 1] = \
    #     X_train_csv_scaled_splits[total_client - 1], X_train_csv_scaled_splits[server_client_index]
    #     X_test_csv_scaled_splits[server_client_index], X_test_csv_scaled_splits[total_client - 1] = \
    #     X_test_csv_scaled_splits[total_client - 1], X_test_csv_scaled_splits[server_client_index]
    #     Y_train_csv_splits[server_client_index], Y_train_csv_splits[total_client - 1] = Y_train_csv_splits[
    #         total_client - 1], Y_train_csv_splits[server_client_index]
    #     Y_test_csv_splits[server_client_index], Y_test_csv_splits[total_client - 1] = Y_test_csv_splits[
    #         total_client - 1], Y_test_csv_splits[server_client_index]
    #     X_train_1_scaled_splits[server_client_index], X_train_1_scaled_splits[total_client - 1] = \
    #     X_train_1_scaled_splits[total_client - 1], X_train_1_scaled_splits[server_client_index]
    #     X_test_1_scaled_splits[server_client_index], X_test_1_scaled_splits[total_client - 1] = X_test_1_scaled_splits[
    #         total_client - 1], X_test_1_scaled_splits[server_client_index]
    #     Y_train_1_splits[server_client_index], Y_train_1_splits[total_client - 1] = Y_train_1_splits[total_client - 1], \
    #     Y_train_1_splits[server_client_index]
    #     Y_test_1_splits[server_client_index], Y_test_1_splits[total_client - 1] = Y_test_1_splits[total_client - 1], \
    #     Y_test_1_splits[server_client_index]
    #     X_train_2_scaled_splits[server_client_index], X_train_2_scaled_splits[total_client - 1] = \
    #     X_train_2_scaled_splits[total_client - 1], X_train_2_scaled_splits[server_client_index]
    #     X_test_2_scaled_splits[server_client_index], X_test_2_scaled_splits[total_client - 1] = X_test_2_scaled_splits[
    #         total_client - 1], X_test_2_scaled_splits[server_client_index]
    #     Y_train_2_splits[server_client_index], Y_train_2_splits[total_client - 1] = Y_train_2_splits[total_client - 1], \
    #     Y_train_2_splits[server_client_index]
    #     Y_test_2_splits[server_client_index], Y_test_2_splits[total_client - 1] = Y_test_2_splits[total_client - 1], \
    #     Y_test_2_splits[server_client_index]

    # Node image1 data replaced
    # height = 32  # 图像高度
    # width = 32  # 图像宽度
    # for index in [3,4,5,6,7,8,9,10,11,12,13,14]:
    #     shape_train = X_train_1_scaled_splits[index].shape[0]
    #     placed_train = np.random.randint(0, 256, (shape_train, 1, height, width))
    #     shape_test = X_test_1_scaled_splits[index].shape[0]
    #     placed_test = np.random.randint(0, 256, (shape_test, 1, height, width))
    #     shape_val = X_val_1_scaled_splits[index].shape[0]
    #     placed_val = np.random.randint(0, 256, (shape_val, 1, height, width))
    #     X_train_1_scaled_splits[index] = placed_train
    #     X_test_1_scaled_splits[index] = placed_test
    #     X_val_1_scaled_splits[index] = placed_val
    for model_name in model_names:
        if model_name =='tc1c2ResModelV3DataV3AdamWithSCVLost':
            # Node csv data replaced
            for index in [6, 7, 8, 9, 10, 11]:
                shape_train = X_train_csv_scaled_splits[index].shape
                placed_train = np.random.rand(*shape_train)
                shape_test = X_test_csv_scaled_splits[index].shape
                placed_test = np.random.rand(*shape_test)
                X_train_csv_scaled_splits[index] = placed_train
                X_test_csv_scaled_splits[index] = placed_test
        elif model_name == 'tc1c2ResModelV3DataV3AdamWithImgLost':
            # Node img data replaced
            for index in [6, 7, 8, 9, 10, 11]:
                shape_train = X_train_1_scaled_splits[index].shape
                placed_train = np.random.rand(*shape_train)
                shape_test = X_test_1_scaled_splits[index].shape
                placed_test = np.random.rand(*shape_test)
                X_train_1_scaled_splits[index] = placed_train
                X_test_1_scaled_splits[index] = placed_test
                shape_train = X_train_2_scaled_splits[index].shape
                placed_train = np.random.rand(*shape_train)
                shape_test = X_test_2_scaled_splits[index].shape
                placed_test = np.random.rand(*shape_test)
                X_train_2_scaled_splits[index] = placed_train
                X_test_2_scaled_splits[index] = placed_test
        for svmethod in svmethods:
            trainValModelCSVIMG(model_name, svmethod, total_client, num_clients, epoch, max_acc,epoch_size,local_epoch_per_round,round_early_stop,
                                X_train_csv_scaled_splits, X_test_csv_scaled_splits,
                                X_train_1_scaled_splits, X_test_1_scaled_splits,
                                X_train_2_scaled_splits, X_test_2_scaled_splits,
                                Y_train_csv_splits, Y_test_csv_splits)
if __name__=="__main__":
    main()