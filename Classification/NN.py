# -*- coding: utf-8 -*-
# @Time : 2019/7/12 上午10:49
# @Author : Sophie_Zhang
# @File : NN.py
# @Software: PyCharm

import os
import time
import torch
import numpy as np
import datetime
from torch import optim, nn, save
from torch import optim, nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn import preprocessing
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

no_cuda = False
device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

def loadData(dataPath, label_path):
    '''
    :param dataPath:
    :param dimension: embedding dimension
    :return: samples, labels
    '''

    samples = np.load(dataPath)

    samples = np.array(samples, dtype=np.float32)

    labels = np.load(label_path)
    labels = np.array(labels, dtype=np.int64)

    features = preprocessing.minmax_scale(samples, axis=0)
    return features, labels

def DataCombine(data, label):
    label = np.array([[i] for i in label], dtype=np.float32)
    return np.concatenate((data, label), axis=1)

class WordDataset(Dataset):
    def __init__(self, data, label):
        self.__data = torch.from_numpy(np.array(data))
        # self.__label = torch.from_numpy(encode_onehot(y.reshape(-1)))
        # self.__label = torch.from_numpy(get_ont_hot(np.array(label), 6))
        # CrossEntropyLoss does not expect a one - hot encoded vector as the target, but class indices
        self.__label = torch.from_numpy(np.array(label))

        # print(self.__label)

    def __getitem__(self, index):
        return self.__data[index], self.__label[index]

    def __len__(self):
        return self.__data.shape[0]

def Acc(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

class MLP(nn.Module):
    def __init__(self, inputsize, outputsize):
        super(MLP, self).__init__()
        self.input = inputsize
        self.output = outputsize
        self.fc1 = nn.Linear(self.input, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.output)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # return F.log_softmax(self.fc3(x), dim=1)
        return self.fc3(x)


def train(X_train, X_test, y_train, y_test, modelpath, n_label, model_index, learning_rate=0.01, epoches=100):
    training_data = DataLoader(dataset=WordDataset(X_train, y_train),
                               batch_size=100,
                               shuffle=True)

    test_data = DataLoader(dataset=WordDataset(X_test, y_test),
                           batch_size=1,
                           shuffle=True)

    model = MLP(X_train.shape[1], n_label).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lossfunc = nn.CrossEntropyLoss().to(device)

    print("\nBegin training...")
    lowest_loss = 10000
    stop = 0
    best_state = model.state_dict()
    break_epoch = 0
    epoch = 0
    for epoch in range(epoches):
        running_loss = 0
        for i, data in enumerate(training_data):
            optimizer.zero_grad()

            (inputs, labels) = data

            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)

            outputs = model(inputs)

            loss = lossfunc(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        avg_loss = running_loss / float(len(training_data))
        print("Epoch:{}, Loss: {}".format(epoch, avg_loss))

        if avg_loss < lowest_loss:
            lowest_loss = avg_loss
            best_state = model.state_dict()
            stop = 0
        else:
            stop += 1

        if stop == 10:
            break_epoch = epoch
            save(model.state_dict(),
                 modelpath + "mlp_model_{}_{}_{}.pt".format(break_epoch, learning_rate, model_index))
            break

    if epoch == epoches - 1:
        save(best_state, modelpath + "mlp_model_{}_{}_{}.pt".format(break_epoch, learning_rate, model_index))

    print("\nBegin prediction...")
    acc_list = []
    model = MLP(X_train.shape[1], n_label).to(device)

    model.load_state_dict(
        torch.load(modelpath + "mlp_model_{}_{}_{}.pt".format(break_epoch, learning_rate, model_index)))

    predict_label = []

    for i, (inputs, labels) in enumerate(test_data):
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        outputs = model(inputs)

        predict_label.append(np.array(torch.argmax(outputs).cpu()).tolist())
        acc_list.append(Acc(outputs, labels))

    print("{}\tFeature index:{}, The Acc. of prediction is: {}".format(datetime.datetime.now(), model_index,
                                                                       sum(acc_list) / len(acc_list)))

    return sum(acc_list) / len(acc_list), predict_label

def kFoldDivid(features, labels, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=2019)
    kf.get_n_splits(features)

    kFold_list = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        kFold_list.append([X_train, X_test, y_train, y_test])

    return kFold_list

def trainAndValidation(corpus):
    path = "features/{}/mix".format(corpus)
    if corpus == "chinese":
        label_path = "label_cn.npy"
        n_label = 6
    elif corpus == "german":
        label_path = "label_de.npy"
        n_label = 3
    else:
        label_path = "label.npy"
        n_label = 6

    model_path = "model_{}/".format(corpus)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    learning_rate = 0.01
    epoches = 200

    print("nytimes:\nlearning_rate:{}\tepoches:{}\n".format(learning_rate, epoches))

    # dimensions = [150, 200, 250, 300]
    # windows = [10]

    dimensions = [300]
    windows = [10]
    #
    # dimensions = [768, 1024]
    # windows = ["offical", corpus]

    dimensions = [768]
    windows = ["offical"]

    respath = "res_{}/".format(corpus)
    if not os.path.exists(respath):
        os.makedirs(respath)

    for dimension in dimensions:
        for window in windows:
            if not (dimension == 1024 and window == corpus):
                print("{}_{} begins...".format(dimension, window))
                feature_path = os.path.join(path, "{}_{}.npy".format(dimension, window))

                features, labels = loadData(feature_path, label_path)
                X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                                    test_size=0.1, random_state=2019,
                                                                    stratify=labels)

                acc_res = 0.0
                acc_list = []
                for i in range(10):
                    acc, predict = train(X_train, X_test, y_train, y_test,
                                         model_path, n_label, i, learning_rate, epoches)
                    print("Acc. in {} turn: {}".format(i, acc))
                    acc_res += acc
                    acc_list.append(acc)

                    np.save(os.path.join(respath, "{}_{}_{}_{}.npy".format(dimension, window, i, acc)), predict)


                print("Corpus: {}\tDimension:{} window:{}, Average acc. of top ten runs:{}".
                      format(corpus, dimension, window, np.sum(acc_list) / 10.0))

                kFold_list = kFoldDivid(X_train, y_train, 10)
                validation_acc = 0.0
                for i, oneFold in enumerate(kFold_list):
                    [X_train, X_test, y_train, y_test] = oneFold
                    acc, predict = train(X_train, X_test, y_train, y_test,
                                         model_path, n_label, i, learning_rate, epoches)
                    validation_acc += acc

                print("Corpus: {}\tDimension:{} window:{}, 10-fold validation acc.:{}".
                      format(corpus, dimension, window, validation_acc / 10.0))

if __name__ == '__main__':
    corpora = ["gutenberg", "nytimes", "E1E2"]
    # corpora = ["chinese"]
    for corpus in corpora:
        trainAndValidation(corpus)