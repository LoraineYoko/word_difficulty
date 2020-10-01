# -*- coding: utf-8 -*-
# @Time : 2019/4/13 下午11:16
# @Author : Sophie_Zhang
# @File : featureExtraction.py
# @Software: PyCharm

'''
feature1:freq
feature2:length
feature3:pronounciation length / vector
feature4:bigram probability_all [float]
feature5:trigram probability_all    [float]
feature6:word embedding [float]
feature7:xpos   [0/1]
feature8:bigram [0/1]
feature9:trigram   [0/1]
feature10:type dependency [0/1]
'''

import os
from math import log
import numpy as np

def zipf_w(freq):
    return log((float(freq)+1.0), 10.0)

def vectorZipf_w(matrix):
    [rows, cols] = matrix.shape
    new_matrix = matrix
    for i in range(rows):
        for j in range(cols):
            new_matrix.itemset((i, j), zipf_w(matrix[i, j]))
    return new_matrix

def vectorFloat(matrix):
    [rows, cols] = matrix.shape
    new_matrix = matrix
    for i in range(rows):
        for j in range(cols):
            new_matrix.itemset((i, j), float(matrix[i, j]))
    return new_matrix

def loadData(path):
    return np.load(path)

def fixedFeatureExtraction(dir, dir_1, embedding_dimension, window):
    freq_path = os.path.join(dir, "freq_no_lemma.npy")
    feature_1 = loadData(freq_path)
    feature_1 = np.array([[i] for i in feature_1])

    length_path = os.path.join(dir, "len.npy")
    feature_2 = loadData(length_path)
    feature_2 = np.array([[float(i)] for i in feature_2])

    pronounciation_path = os.path.join(dir, "pronunciation_binary.npy")
    feature_3 = loadData(pronounciation_path)
    # feature_7 = vectorFloat(feature_3)

    bigram_proba_path = os.path.join(dir, "bigram_proba_all.npy")
    feature_4 = loadData(bigram_proba_path)
    feature_4 = np.array([[i] for i in feature_4])

    trigram_proba_path = os.path.join(dir, "trigram_proba_all.npy")
    feature_5 = loadData(trigram_proba_path)
    feature_5 = np.array([[i] for i in feature_5])

    embedding_path = os.path.join(dir_1, "embedding_{}_{}.npy".format(embedding_dimension, window))
    feature_6 = loadData(embedding_path)

    return feature_1, feature_2, feature_3, feature_4, feature_5, feature_6

def binaryFeatureExtraction(dir):
    # upos_path = os.path.join(dir, "upos_binary.npy")
    # feature_7 = loadData(upos_path)
    # feature_7 = vectorFloat(feature_7)

    xpos_path = os.path.join(dir, "xpos_binary.npy")
    feature_8 = loadData(xpos_path)
    feature_8 = vectorFloat(feature_8)

    bigram_path = os.path.join(dir, "bigram_binary.npy")
    feature_9 = loadData(bigram_path)
    feature_9 = vectorFloat(feature_9)

    trigram_path = os.path.join(dir, "trigram_binary.npy")
    feature_10 = loadData(trigram_path)
    feature_10 = vectorFloat(feature_10)

    dependency_path = os.path.join(dir, "type_binary.npy")
    feature_11 = loadData(dependency_path)
    feature_11 = vectorFloat(feature_11)

    return feature_8, feature_9, feature_10, feature_11

def freqFeatureExtraction(dir):
    # upos_path = os.path.join(dir, "upos_freq.npy")
    # feature_7 = loadData(upos_path)
    # feature_7 = vectorZipf_w(feature_7)

    xpos_path = os.path.join(dir, "xpos_binary.npy")
    feature_8 = loadData(xpos_path)
    feature_8 = vectorZipf_w(feature_8)

    bigram_path = os.path.join(dir, "bigram_freq.npy")
    feature_9 = loadData(bigram_path)
    feature_9 = vectorFloat(feature_9)

    trigram_path = os.path.join(dir, "trigram_freq.npy")
    feature_10 = loadData(trigram_path)
    feature_10 = vectorFloat(feature_10)

    dependency_path = os.path.join(dir, "type_freq.npy")
    feature_11 = loadData(dependency_path)
    feature_11 = vectorZipf_w(feature_11)

    return feature_8, feature_9, feature_10, feature_11


def featureExtraction(dir, dir_1, des_dir, embedding_dimension, window):
    feature_1, feature_2, feature_3, feature_4, feature_5, feature_6 = fixedFeatureExtraction(dir, dir_1, embedding_dimension, window)
    # print(feature_1)
    # binary feature
    feature_8, feature_9, feature_10, feature_11 = binaryFeatureExtraction(dir)

    feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_8, feature_9, feature_10, feature_11]

    for i, feature in enumerate(feature_list):
        print("feature_{}:{}".format(i, feature.shape))

    des_txt_path = os.path.join(des_dir, "binary_{}dimension.txt".format(embedding_dimension))
    des_npy_path = os.path.join(des_dir, "glove_{}_{}.npy".format(embedding_dimension, window))
    features = np.concatenate((feature_1, feature_2, feature_3, feature_4,
                               feature_5, feature_6, feature_8,
                               feature_9, feature_10, feature_11), axis=1)


    # with open(des_txt_path, "w") as fw:
    #     print(features.shape)
    #     [rows, cols] = features.shape
    #     for i in range(rows):
    #         for j in range(cols):
    #             fw.write("{}\t".format(features[i, j]))
    #         fw.write("\n")

    np.save(des_npy_path, features)
    print("feature dimension:{} window:{} Ok".format(dimension, window))

    # # freq feature
    # feature_7, feature_8, feature_9, feature_10, feature_11 = freqFeatureExtraction(dir)
    #
    # feature_list = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6,
    #                 feature_7, feature_8, feature_9, feature_10, feature_11]
    #
    # for i, feature in enumerate(feature_list):
    #     print("feature_{}:{}".format(i, feature.shape))
    #
    # des_txt_path = os.path.join(des_dir, "freq_{}dimension.txt".format(embedding_dimension))
    # des_npy_path = os.path.join(des_dir, "freq_{}dimension.npy".format(embedding_dimension))
    # features = np.concatenate((feature_1, feature_2, feature_3, feature_4,
    #                            feature_5, feature_6, feature_7, feature_8,
    #                            feature_9, feature_10, feature_11), axis=1)
    #
    # with open(des_txt_path, "w") as fw:
    #     [rows, cols] = features.shape
    #     for i in range(rows):
    #         for j in range(cols):
    #             fw.write("{}\t".format(features[i, j]))
    #         fw.write("\n")
    #
    # np.save(des_npy_path, features)


def wraper(dir, dir_1, despath, embedding_dimension, window):
    feature_1, feature_2, feature_3, feature_4, feature_5, feature_6 = fixedFeatureExtraction(dir, dir_1, embedding_dimension, window)
    feature_8, feature_9, feature_10, feature_11 = binaryFeatureExtraction(dir)

    featureList = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6,
                   feature_8, feature_9, feature_10, feature_11]

    for i, feature in enumerate(featureList):
        # print()
        samples = np.concatenate(tuple(featureList[j] for j in range(len(featureList)) if not i==j), axis=1)
        # with open(os.path.join(despath, "feature-{}.txt".format(i)), "w") as fw:
        #     [rows, cols] = samples.shape
        #     for m in range(rows):
        #         for n in range(cols):
        #             fw.write("{}\t".format(samples[m, n]))
        #         fw.write("\n")

        np.save(os.path.join(despath, "feature-{}-{}-{}.npy".format(embedding_dimension, window, i)), samples)
        print("Wrapper dimension:{} window:{} -{} ok.".format(dimension, window, i))

def labelExtraction():
    word_level_dict = {}
    level_list = ["A1", "A2", "B1", "B2", "C1", "C2"]
    label = 0
    for level in level_list:
        path = os.path.join("wordlist_ignore_overlap_cmudict_filter", "{}.txt".format(level))
        with open(path, "r") as fr:
            for line in fr:
                word = line.strip()
                word_level_dict[word] = str(label)
        label += 1

    label_list = []
    with open("wordlist.txt", "r") as f:
        for line in f:
            word = line.strip()
            label_list.append(word_level_dict[word])

    with open("label.txt", "w") as fw:
        for label in label_list:
            fw.write(label + "\n")

    np.save("label.npy", label_list)

if __name__ == '__main__':
    # corpus = "nytimes"
    corpus = "chinese"

    dir = "features/{}/single_feature".format(corpus)
    des_dir_w = "features/{}/wrapper_feature/".format(corpus)

    if not os.path.exists(des_dir_w):
        os.makedirs(des_dir_w)

    embedding_dimensions = [300]
    windows = [5]
    for dimension in embedding_dimensions:
        for window in windows:
            featureExtraction(dir, dir, des_dir_w, dimension, window)
            wraper(dir, dir, des_dir_w, dimension, window)
    # labelExtraction()
    print("Finished.")
