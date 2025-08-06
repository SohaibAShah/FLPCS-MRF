#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import re 
import numpy as np
import cv2
from zipfile import ZipFile
import shutil
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def read_data(data):
    data.reset_index(inplace = True)
    data.rename(columns={'level_0': 'Time'}, inplace = True)
    data.rename(columns={'x-axis (deg/s).4': 'Raw Brainwave Signal '}, inplace = True)
    data.rename(columns={'Unnamed: 42': 'Tag'}, inplace = True)
    
    TimeStamp = data.iloc[:,0]
    ankle = data.iloc[: , 1:8]
    pocket = data.iloc[:, 8:15]
    waist = data.iloc[:,15:22]
    neck = data.iloc[:,22:29]
    wrist = data.iloc[:,29:36]
    EEG = data.iloc[:,36]
    Infraded = data.iloc[:,37:43]
    label = data.iloc[:,46]
        
    ankle.columns = ['X-axis Accelerometer (g)', 'Y-axis Accelerometer (g)' , 'Z-axis Accelerometer (g)',
                     'Roll Gyroscrope (deg/s)', 'Pitch Gyroscope (deg/s)', 'Yaw Gyroscope (deg/s)' ,'Luminosity (lux)']

    pocket.columns = ['X-axis Accelerometer (g)', 'Y-axis Accelerometer (g)' , 'Z-axis Accelerometer (g)',
                     'Roll Gyroscrope (deg/s)', 'Pitch Gyroscope (deg/s)', 'Yaw Gyroscope (deg/s)' ,'Luminosity (lux)']

    waist.columns = ['X-axis Accelerometer (g)', 'Y-axis Accelerometer (g)' , 'Z-axis Accelerometer (g)',
                     'Roll Gyroscrope (deg/s)', 'Pitch Gyroscope (deg/s)', 'Yaw Gyroscope (deg/s)' ,'Luminosity (lux)']

    neck.columns = ['X-axis Accelerometer (g)', 'Y-axis Accelerometer (g)' , 'Z-axis Accelerometer (g)',
                     'Roll Gyroscrope (deg/s)', 'Pitch Gyroscope (deg/s)', 'Yaw Gyroscope (deg/s)' ,'Luminosity (lux)']

    wrist.columns = ['X-axis Accelerometer (g)', 'Y-axis Accelerometer (g)' , 'Z-axis Accelerometer (g)',
                     'Roll Gyroscrope (deg/s)', 'Pitch Gyroscope (deg/s)', 'Yaw Gyroscope (deg/s)' ,'Luminosity (lux)']

    Infraded.columns = ['Infrared 1', 'Infrared 2', 'Infrared 3', 'Infrared 4', 'Infrared 5', 'Infrared 6']

    handled_data = pd.concat([TimeStamp, ankle,pocket,waist,neck,wrist,EEG,Infraded ], 
                            axis = 1, 
                            keys = ['TimeStamp','Wearable Ankle', 'Wearable Pocket','Wearable Waist', 
                                    'Wearable Neck', 'Wearable Wrist','EEG Headset'  ,'Infrared'],
                             
                            names = ['Deviece Name', 'Channel Name'])
    handled_data[('Tag' , 'Label')]= label
    
    return handled_data


def concat_data_train(subIn,data_path):
    concat_Sub = []
    list_Sub = []
    sum_shape = 0
    sum_csv = 0
    Sub = 'Subject' + str(subIn)
    list_Sub.append(Sub)
    concat_Act = []
    list_Act = []
    for act_ in range(1,11+1):
        Act = 'Activity'+ str(act_)
        list_Act.append(Act)
        concat_Trial = []
        list_Trial  = []
        for trial_ in range(1,3):
            Trial = 'Trial'+ str(trial_)
            list_Trial.append(Trial)
            path = '/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_sensor_files/' + Sub + Act + Trial + '.csv'
            data = pd.read_csv(path,skiprows=1)
            print('path : {} . Shape : ({},{})'.format(path, data.shape[0], data.shape[1]))
            sum_shape += data.shape[0]
            sum_csv +=1
            handled  = read_data(data)
            concat_Trial.append(handled)
        TRIAL = pd.concat(concat_Trial,keys = list_Trial)
        concat_Act.append(TRIAL)
    ACT = pd.concat(concat_Act, keys = list_Act)
    concat_Sub.append(ACT)
    SUB = pd.concat(concat_Sub,keys = list_Sub)
    return SUB

def concat_data_test(subIn,data_path):
    concat_Sub = []
    list_Sub = []
    sum_shape = 0
    sum_csv = 0
    Sub = 'Subject' + str(subIn)
    list_Sub.append(Sub)
    concat_Act = []
    list_Act = []
    for act_ in range(1,11+1):
        Act = 'Activity'+ str(act_)
        list_Act.append(Act)
        concat_Trial = []
        list_Trial  = []
        trial_ = 3
        Trial = 'Trial' + str(trial_)
        list_Trial.append(Trial)
        path = '/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_sensor_files/' + Sub + Act + Trial + '.csv'
        data = pd.read_csv(path, skiprows=1)
        print('path : {} . Shape : ({},{})'.format(path, data.shape[0], data.shape[1]))
        sum_shape += data.shape[0]
        sum_csv += 1
        handled = read_data(data)
        concat_Trial.append(handled)
        TRIAL = pd.concat(concat_Trial,keys = list_Trial)
        concat_Act.append(TRIAL)
    ACT = pd.concat(concat_Act, keys = list_Act)
    concat_Sub.append(ACT)
    SUB = pd.concat(concat_Sub,keys = list_Sub)
    return SUB

def load_img_train(data_path, sub_,start_act, end_act,  start_cam,  end_cam , DesiredWidth = 64, DesiredHeight = 64):
    IMG = []
    count = 0
    name_img = []
    sub = 'Subject' + str(sub_)
    for act_ in range(start_act, end_act + 1):
        act = 'Activity' + str(act_)

        for trial_ in range(1, 3):
            trial = 'Trial' + str(trial_)
            for cam_ in range(start_cam, end_cam + 1):
                cam = 'Camera' + str(cam_)
                try:
                    with ZipFile(
                            '/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_camera_files/' + sub + act + trial + cam + '.zip',
                            'r') as zipObj:
                        zipObj.extractall('CAMERA/' + sub + act + trial + cam)
                except Exception as result:
                    print('/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_camera_files/' + sub + '/' + act + '/' + trial + '/' + sub + act + trial + cam + '.zip',
                          result)

                for root, dirnames, filenames in os.walk('CAMERA/' + sub + act + trial + cam):
                    for filename in filenames:
                        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                            filepath = os.path.join(root, filename)
                            count += 1
                            if count % 5000 == 0:
                                print('{} : {} '.format(filepath, count))
                            if filepath == 'CAMERA/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png':
                                print('----------------------------NO SHAPE---------------------------------')
                                continue
                            elif len(filepath) > 70:
                                print(' {} : Invalid image'.format(filepath))
                                continue
                            name_img.append(filepath)
                            img = cv2.imread(filepath, 0)
                            resized = ResizeImage(img, DesiredWidth, DesiredHeight)
                            IMG.append(resized)
                shutil.rmtree('CAMERA/' + sub + act + trial + cam)

    return IMG , name_img


def load_img_test(data_path, sub_, start_act, end_act, start_cam, end_cam, DesiredWidth=64, DesiredHeight=64):
    IMG = []
    count = 0
    name_img = []
    sub = 'Subject' + str(sub_)
    for act_ in range(start_act, end_act + 1):
        act = 'Activity' + str(act_)
        trial_ = 3
        trial = 'Trial' + str(trial_)
        for cam_ in range(start_cam, end_cam + 1):
            cam = 'Camera' + str(cam_)
            try:
                with ZipFile(
                        '/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_camera_files/' + sub + act + trial + cam + '.zip',
                        'r') as zipObj:
                    zipObj.extractall('CAMERA/' + sub + act + trial + cam)
            except Exception as result:
                print('/home/syed/PhD/Fall-Detection-Research-1/UP-Fall Dataset/downloaded_camera_files/' + sub + act + trial + cam + '.zip',
                      result)

            for root, dirnames, filenames in os.walk('CAMERA/' + sub + act + trial + cam):
                for filename in filenames:
                    if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                        filepath = os.path.join(root, filename)
                        count += 1
                        if count % 5000 == 0:
                            print('{} : {} '.format(filepath, count))
                        if filepath == 'CAMERA/Subject6Activity10Trial2Camera2/2018-07-06T12_03_04.483526.png':
                            print('----------------------------NO SHAPE---------------------------------')
                            continue
                        elif len(filepath) > 70:
                            print(' {} : Invalid image'.format(filepath))
                            continue
                        name_img.append(filepath)
                        img = cv2.imread(filepath, 0)
                        resized = ResizeImage(img, DesiredWidth, DesiredHeight)
                        IMG.append(resized)
            shutil.rmtree('CAMERA/' + sub + act + trial + cam)
    return IMG, name_img


def handle_name(path_name) :
    img_name = []
    for path in path_name :
        if len(path) == 68: 
            img_name.append(path[38:64])
        elif len(path) == 69 :
            img_name.append(path[39:65])
        else :
            img_name.append(path[40:66])
    handle = []
    for name in img_name :
        n1 = 13
        a1 = name.replace(name[n1],':')
        n2 = 16
        a2 = a1.replace(name[n2],':')
        handle.append(a2)
    return handle 


def ShowImage(ImageList, nRows = 1, nCols = 2, WidthSpace = 0.00, HeightSpace = 0.00):
    gs = gridspec.GridSpec(nRows, nCols)     
    gs.update(wspace=WidthSpace, hspace=HeightSpace) # set the spacing between axes.
    plt.figure(figsize=(20,20))
    for i in range(len(ImageList)):
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        plt.subplot(nRows, nCols,i+1)
        image = ImageList[i].copy()
        if (len(image.shape) < 3):
            plt.imshow(image, plt.cm.gray)
        else:
            plt.imshow(image)
        plt.title("Image " + str(i))
        plt.axis('off')
    plt.show()
    
    
def ResizeImage(IM, DesiredWidth, DesiredHeight):
    OrigWidth = float(IM.shape[1])
    OrigHeight = float(IM.shape[0])
    Width = DesiredWidth 
    Height = DesiredHeight

    if((Width == 0) & (Height == 0)):
        return IM
    
    if(Width == 0):
        Width = int((OrigWidth * Height)/OrigHeight)

    if(Height == 0):
        Height = int((OrigHeight * Width)/OrigWidth)

    dim = (Width, Height)
    resizedIM = cv2.resize(IM, dim, interpolation = cv2.INTER_NEAREST) 
    return resizedIM

def main():
    # path
    sensor_path = './dataset'
    data_path = './dataset/UP-Fall-Detection'
    camera_path = './dataset/camera'
    # camera_path = 'D://PythonProjects//MutilModelDataAnalyseForHealth//FedFall//HAR-UP-master//data//UP-Fall-Detection-Unzip//'
    # subs = [1,2,3,4,6,7,8,10]
    # subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    subs = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]

    for sub in subs:

        ### processing sensor data
        # sensor_data = pd.read_csv(os.path.join(sensor_path, 'CompleteDataSet.csv'), skiprows=2, header=None)

        SUB_train = concat_data_train(sub,data_path)
        SUB_test = concat_data_test(sub,data_path)

        SUB_train.to_csv(path_or_buf='./dataset/Sensor + Image/{}_sensor_train.csv'.format(sub), index=False)
        SUB_test.to_csv(path_or_buf='./dataset/Sensor + Image/{}_sensor_test.csv'.format(sub), index=False)

        times_train = SUB_train.iloc[:, 0].values
        labels_train = SUB_train.iloc[:, -1].values
        Time_Label_train = pd.DataFrame(labels_train, index=times_train)
        print('{}_train_Time_Label.shape'.format(sub),Time_Label_train.shape)

        times_test = SUB_test.iloc[:, 0].values
        labels_test  = SUB_test.iloc[:, -1].values
        Time_Label_test  = pd.DataFrame(labels_test, index=times_test)
        print('{}_test_Time_Label.shape'.format(sub), Time_Label_test.shape)

        # ## Load image
        start_act = 1
        end_act = 11
        start_cam = 1
        end_cam = 1
        DesiredWidth = 32
        DesiredHeight = 32

        img_1_train, path_1_train = load_img_train(data_path,sub,
                                 start_act, end_act,
                                 start_cam, end_cam, DesiredWidth, DesiredHeight)

        name_1_train = handle_name(path_1_train)

        img_1_test, path_1_test = load_img_test(data_path, sub,
                                                   start_act, end_act,
                                                   start_cam, end_cam, DesiredWidth, DesiredHeight)

        name_1_test = handle_name(path_1_test)

        start_act = 1
        end_act = 11
        start_cam = 2
        end_cam = 2

        DesiredWidth = 32
        DesiredHeight = 32

        img_2_train, path_2_train = load_img_train(data_path, sub,
                                                   start_act, end_act,
                                                   start_cam, end_cam, DesiredWidth, DesiredHeight)

        name_2_train = handle_name(path_2_train)

        img_2_test, path_2_test = load_img_test(data_path, sub,
                                                start_act, end_act,
                                                start_cam, end_cam, DesiredWidth, DesiredHeight)

        name_2_test = handle_name(path_2_test)

        print('len(img_1_train)', len(img_1_train))
        print('len(name_1_train)', len(name_1_train))
        print('len(img_1_test)', len(img_2_test))
        print('len(name_1_test)', len(name_2_test))
        print('len(img_2_train)', len(img_2_train))
        print('len(name_2_train)', len(name_2_train))
        print('len(img_2_test)', len(img_2_test))
        print('len(name_2_test)', len(name_2_test))

        # ind1 = np.arange(0, 294678)
        ind1_train = np.arange(0, len(img_1_train))
        red_in1_train = ind1_train[~np.isin(name_1_train, name_2_train)]

        name_1d_train = np.delete(name_1_train, red_in1_train)
        img_1d_train = np.delete(img_1_train, red_in1_train, axis=0)

        # ind2 = np.arange(0, 294678)
        ind2_train = np.arange(0, len(img_2_train))
        red_in2_train = ind2_train[~np.isin(name_2_train, name_1_train)]

        name_2d_train = np.delete(name_2_train, red_in2_train)
        img_2d_train = np.delete(img_2_train, red_in2_train, axis=0)

        print('name_1d_train == name_2d_train).all():',(name_1d_train == name_2d_train).all())

        label_1_train = Time_Label_train.loc[name_1d_train].values
        label_2_train = Time_Label_train.loc[name_2d_train].values

        print('len(img_1d_train)',len(img_1d_train))
        print('len(name_1d_train)',len(name_1d_train))
        print('len(label_1_train)',len(label_1_train))
        print('len(img_2d_train)',len(img_2d_train))
        print('len(name_2d_train)',len(name_2d_train))
        print('len(label_2_train)',len(label_2_train))

        cam = '1'

        image = './dataset/Sensor + Image' + '/' + '{}_image_1_train.npy'.format(sub)
        name = './dataset/Sensor + Image' + '/' + '{}_name_1_train.npy'.format(sub)
        label = './dataset/Sensor + Image' + '/' + '{}_label_1_train.npy'.format(sub)

        np.save(image, img_1d_train)
        np.save(name, name_1d_train)
        np.save(label, label_1_train)

        cam = '2'
        image = './dataset/Sensor + Image' + '/' + '{}_image_2_train.npy'.format(sub)
        name = './dataset/Sensor + Image' + '/' + '{}_name_2_train.npy'.format(sub)
        label = './dataset/Sensor + Image' + '/' + '{}_label_2_train.npy'.format(sub)

        np.save(image, img_2d_train)
        np.save(name, name_2d_train)
        np.save(label, label_2_train)

        ind1_test = np.arange(0, len(img_1_test))
        red_in1_test = ind1_test[~np.isin(name_1_test, name_2_test)]

        name_1d_test = np.delete(name_1_test, red_in1_test)
        img_1d_test = np.delete(img_1_test, red_in1_test, axis=0)

        # ind2 = np.arange(0, 294678)
        ind2_test = np.arange(0, len(img_2_test))
        red_in2_test = ind2_test[~np.isin(name_2_test, name_1_test)]

        name_2d_test = np.delete(name_2_test, red_in2_test)
        img_2d_test = np.delete(img_2_test, red_in2_test, axis=0)

        print('(name_1d_test == name_2d_test).all():', (name_1d_test == name_2d_test).all())


        label_1_test = Time_Label_test.loc[name_1d_test].values
        label_2_test = Time_Label_test.loc[name_2d_test].values

        print('len(img_1d_test)', len(img_1d_test))
        print('len(name_1d_test)', len(name_1d_test))
        print('len(label_1_test)', len(label_1_test))
        print('len(img_2d_test)', len(img_2d_test))
        print('len(name_2d_test)', len(name_2d_test))
        print('len(label_2_test)', len(label_2_test))

        cam = '1'

        image = './dataset/Sensor + Image' + '/' + '{}_image_1_test.npy'.format(sub)
        name = './dataset/Sensor + Image' + '/' + '{}_name_1_test.npy'.format(sub)
        label = './dataset/Sensor + Image' + '/' + '{}_label_1_test.npy'.format(sub)

        np.save(image, img_1d_test)
        np.save(name, name_1d_test)
        np.save(label, label_1_test)

        cam = '2'
        image = './dataset/Sensor + Image' + '/' + '{}_image_2_test.npy'.format(sub)
        name = './dataset/Sensor + Image' + '/' + '{}_name_2_test.npy'.format(sub)
        label = './dataset/Sensor + Image' + '/' + '{}_label_2_test.npy'.format(sub)

        np.save(image, img_2d_test)
        np.save(name, name_2d_test)
        np.save(label, label_2_test)

if __name__=="__main__":
    main()