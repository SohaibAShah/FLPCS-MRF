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


# Define dataset loader
class CostumDataset(Dataset):
    def __init__(self, features1, features2, labels):
        self.features1 = features1
        self.features2 = features2
        self.labels = labels

    def __len__(self):
        return len(self.features1)
    
    def __getitem__(self, index):
        return self.features1[index], self.features2[index], self.labels[index]


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


class ModelCSVIMG1(nn.Module):
    def __init__(self, num_csv_features):
        super(ModelCSVIMG1, self).__init__()

        # v2==========================================
        # 第一输入分支：处理CSV特征
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_fc_3 = nn.Linear(600, 100)
        self.csv_dropout = nn.Dropout(0.2)

        # 第二输入分支：处理第一张图像的2D卷积
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # 全连接层
        self.fc1 = nn.Linear(200, 600)
        self.fc2 = nn.Linear(600, 1200)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1200, 12)

    def forward(self, x_csv, x_img1):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = F.relu(self.csv_fc_3(x_csv))
        x_csv = self.csv_dropout(x_csv)
        # x_csv = self.fc_csv_3(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        # 第二分支：第一张图像
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.view(x_img1.size(0), -1)
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 =self.img1_dropout(x_img1)

        # 连接三个分支
        x = torch.cat((x_csv, x_img1), dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=1)

        return x


def trainValModelCSVIMG1(total_client,num_clients,epoch,max_acc,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits):
    # Instantiate the model    the total_client’th split used for server
    # input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    model_MLP = ModelCSVIMG1(X_train_csv_scaled_splits[total_client-1].shape[1])
    model_MLP = model_MLP.double()
    model_MLP = model_MLP.cuda()

    # initialize server and clients
    server = Server(model_MLP, [X_test_csv_scaled_splits[total_client-1],X_test_1_scaled_splits[total_client-1],Y_test_csv_splits[total_client-1]], num_clients)
    clients = []

    for c in range(total_client):
        clients.append(Client(server.global_model, [X_train_csv_scaled_splits[c],X_train_1_scaled_splits[c],Y_train_csv_splits[c]],
                              [X_val_csv_scaled_splits[c],X_val_1_scaled_splits[c],Y_val_csv_splits[c]], id=c))

    # train
    clients_acc = {}
    clients_loss = {}
    for i in range(num_clients + 1):  # one more for server
        epoch_acc = []
        epoch_loss = []
        clients_acc[i] = epoch_acc
        clients_loss[i] = epoch_loss
    for e in range(epoch):
        candidates = random.sample(clients, num_clients)  # randomly select clients
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # initialize weight_accumulator
        client_index = 0
        for c in candidates:
            diff, acc_client, loss_client = c.local_train(server.global_model)  # train local model
            clients_acc[client_index].append(acc_client)
            clients_loss[client_index].append(loss_client)
            client_index += 1
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])  # update weight_accumulator
        server.model_aggregate(weight_accumulator)  # aggregate global model
        acc, loss = server.model_eval()
        clients_acc[num_clients].append(acc_client)
        clients_loss[num_clients].append(loss_client)
        print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))
        if acc > max_acc:
            max_acc = acc
            torch.save(server.global_model.state_dict(), "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
            print("save model")

    # test
    model = model_MLP
    model.load_state_dict(torch.load("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))))
    model.eval()
    model = model.cuda()

    y_test, y_predict = [], []

    test_loader = torch.utils.data.DataLoader(CostumDataset(X_val_csv_scaled_splits[total_client-1],X_val_1_scaled_splits[total_client-1],Y_val_csv_splits[total_client-1]), batch_size=32)

    for batch_id, batch in enumerate(test_loader):
        data1 = batch[0]
        data2 = batch[1]
        # target = torch.squeeze(batch[3]).int()
        # target = torch.tensor(target, dtype=torch.int64)
        target = torch.squeeze(batch[2]).float()

        if torch.cuda.is_available():
            data1 = data1.cuda()
            data2 = data2.cuda()
            target = target.cuda()

        output = model(data1,data2)
        y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
        y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())

    # classification_report
    print(classification_report(y_test, y_predict,
                                target_names=[f'A{i}' for i in range(1, 13)], digits=4))

    print('max_acc',max_acc)
    csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.csv".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    # 保存到 CSV 文件
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['client', 'Epoch', 'Loss', 'Accuracy'])
        for i in range(num_clients + 1):
            # 添加列名
            losses = clients_loss[i]
            accuracies = clients_acc[i]
            for j in range(epoch):
                writer.writerow([i, j + 1, losses[j], accuracies[j]])

    # confusion matrix
    plt.figure(dpi=150, figsize=(6, 4))
    classes = [f'A{i}' for i in range(1, 13)]
    mat = confusion_matrix(y_test, y_predict)

    df_cm = pd.DataFrame(mat, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    # 保存图像
    plt.savefig("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    plt.show()


def main():
    set_seed()
    # hyperparameters
    max_acc = 80  # thorshold of accuracy (80%), for saving best model
    epoch = 200
    total_client = 15  # total number of clients
    num_clients = 12  # number of clients selected per round
    # 定义比例，默认均分
    ratios = [1 / total_client] * total_client
    # load data
    X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled, \
        Y_train_csv, Y_test_csv, Y_val_csv, \
        X_train_1_scaled, X_test_1_scaled, X_val_1_scaled, \
        Y_train_1, Y_test_1, Y_val_1, \
        X_train_2_scaled, X_test_2_scaled, X_val_2_scaled, \
        Y_train_2, Y_test_2, Y_val_2 = loadData()
    # split data according to total_client
    X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
        Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
        X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
        Y_train_2_splits, Y_test_2_splits, Y_val_2_splits = splitForClients(total_client,ratios,X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled,
        Y_train_csv, Y_test_csv, Y_val_csv,
        X_train_1_scaled, X_test_1_scaled, X_val_1_scaled,
        Y_train_1, Y_test_1, Y_val_1,
        X_train_2_scaled, X_test_2_scaled, X_val_2_scaled,
        Y_train_2, Y_test_2, Y_val_2 )

    trainValModelCSVIMG1(total_client, num_clients, epoch, max_acc,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits)


# Server
class Server(object):
    def __init__(self, model, eval_dataset, num_clients):
        
        self.global_model = model
        self.num_clients = num_clients
        self.serverTestDataSet = CostumDataset(eval_dataset[0],eval_dataset[1],eval_dataset[2])
        self.eval_loader = torch.utils.data.DataLoader(self.serverTestDataSet, batch_size=32)
	
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
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[2]).float()
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            if torch.cuda.is_available():
                data1 = data1.cuda()
                data2 = data2.cuda()
                target = target.cuda()
            
            output = self.global_model(data1,data2)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 *(float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        return acc, loss


# Client
class Client(object):
    def __init__(self, model, train_dataset,val_dataset, id = -1):
                self.local_model = model
                self.client_id = id
                self.train_dataset = CostumDataset(train_dataset[0],train_dataset[1],train_dataset[2])
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32)
                self.eval_dataset = CostumDataset(val_dataset[0], val_dataset[1], val_dataset[2])
                self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=32)

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
        self.local_model.train()
        for e in range(3):
            for batch_id, batch in enumerate(self.train_loader):
                data1 = batch[0]
                data2 = batch[1]
                # target = torch.squeeze(batch[1]).int()
                # target = torch.tensor(target, dtype=torch.int64)
                target = torch.squeeze(batch[2]).float()
                # target = torch.tensor(target, dtype=float)

                if torch.cuda.is_available():
                    data1 = data1.cuda()
                    data2 = data2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data1,data2)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        # after epoch train, eval model and save client model acc,loss to csv
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data1 = batch[0]
            data2 = batch[1]
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[2]).float()
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            if torch.cuda.is_available():
                data1 = data1.cuda()
                data2 = data2.cuda()
                target = target.cuda()

            output = self.local_model(data1,data2)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff,acc,loss

model_name = 'tc1Model'
if __name__=="__main__":
    main()