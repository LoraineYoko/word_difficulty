# -*- coding: utf-8 -*-
# @Time : 2019/4/13 下午5:09
# @Author : Sophie_Zhang
# @File : preprocess.py
# @Software: PyCharm

import os
import pickle
import numpy as np
from math import log

def getWordList(path):
    wordlist = []
    with open(path, 'r') as fr:
        for line in fr:
            wordlist.append(line.strip())
    return wordlist

def readNum(file):
    word_num_dict = {}
    with open(file, 'r') as fr:
        for line in fr:
            line = line.strip().split("\t")
            word_num_dict[line[0]] = float(line[1])
    return word_num_dict

def readVector(file):
    word_vector_dict = {}
    vector_len = 0
    with open(file, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            word_vector_dict[line[0].lower()] = [float(i) for i in line[1:]]
            vector_len = len(line[1:])
    return word_vector_dict, vector_len

def Zipf_w(freq):
    return log(float(freq+1), 10)


def FreqProcess(freqPath, featurePath):
    freqDict = readNum(freqPath)
    freq_list = []
    for word in wordlist:
        if not word in freqDict.keys():
            freq_list.append(0.0)
        else:
            freq_list.append(Zipf_w(freqDict[word]))

    # with open(os.path.join(featurePath, "freq_no_lemma.txt"), 'wb') as fw:
    #     for freq in freq_list:
    #         fw.write(("{}\n".format(freq)).encode())

    np.save(os.path.join(featurePath, "freq_no_lemma.npy"), np.array(freq_list))

def posProcess(posPath, featurePath):
    pos_binary_path = os.path.join(posPath, "xpos_vector.txt")
    # pos_freq_path = os.path.join(posPath, "xpos_vector_freq.txt")


    pos_binary_dict, vector_len = readVector(pos_binary_path)
    # pos_freq_dict = readVector(pos_freq_path)

    pos_binary_list, pos_freq_list = [], []
    # print(pos_binary_dict)

    # print(pos_binary_dict[b'meistens'])
    print(pos_binary_dict["画"])

    # pos_binary_len = len(pos_binary_dict[b'meistens'])
    pos_binary_len = len(pos_binary_dict["画"])
    for word in wordlist:
        if not word in pos_binary_dict.keys():
            pos_binary_list.append([0 for i in range(pos_binary_len)])
        else:
            pos_binary_list.append(pos_binary_dict[word])

        # if not word in pos_freq_dict.keys():
        #     pos_freq_list.append([0 for i in range(pos_freq_len)])
        # else:
        #     pos_freq_list.append(pos_freq_dict[word])

    with open(os.path.join(featurePath, "xpos_binary.txt"), 'w') as fw:
        for word in pos_binary_list:
            for pos in word:
                fw.write(("{}\t".format(pos)))
            fw.write("\n")

    np.save(os.path.join(featurePath, "xpos_binary.npy"), np.array(pos_binary_list))

    # with open(os.path.join(featurePath, "xpos_freq.txt"), 'wb') as fw:
    #     for word in pos_freq_list:
    #         for pos in word:
    #             fw.write("{}\t".format(pos).encode())
    #         fw.write(b"\n")

    # np.save(os.path.join(featurePath, "xpos_freq.npy"), np.array(pos_freq_list))

def bigramAndTrigram(nGramPath, featurePath):
    ngram_binary_path = os.path.join(nGramPath, "vector.txt")
    ngram_freq_path = os.path.join(nGramPath, "vector_freq.txt")
    ngram_proba_path = os.path.join(nGramPath, "probability_all_log.txt")

    ngram_binary_dict, vector_len = readVector(ngram_binary_path)
    ngram_freq_dict, vector_len = readVector(ngram_freq_path)
    ngram_proba_dict = readNum(ngram_proba_path)

    ngram_binary_list, ngram_freq_list, ngram_proba_list = [], [], []

    # print(ngram_binary_dict)

    # ngram_binary_len, bigram_freq_len = len(ngram_binary_dict[b"meistens"]), len(ngram_freq_dict[b"meistens"])
    ngram_binary_len, bigram_freq_len = len(ngram_binary_dict["画"]), len(ngram_freq_dict["画"])
    for word in wordlist:
        if not word in ngram_binary_dict.keys():
            ngram_binary_list.append([0 for i in range(ngram_binary_len)])
        else:
            ngram_binary_list.append(ngram_binary_dict[word])

        if not word in ngram_freq_dict.keys():
            ngram_freq_list.append([0 for i in range(bigram_freq_len)])
        else:
            ngram_freq_list.append(ngram_freq_dict[word])

        if not word in ngram_proba_dict.keys():
            ngram_proba_list.append(0.0)
        else:
            ngram_proba_list.append(ngram_proba_dict[word])


    # with open(os.path.join(featurePath, "bigram_binary.txt"), 'wb') as fw:
    #     for word in ngram_binary_list:
    #         for pos in word:
    #             fw.write("{}\t".format(pos))
    #         fw.write(b"\n")

    np.save(os.path.join(featurePath, "bigram_binary.npy"), np.array(ngram_binary_list))

    # with open(os.path.join(featurePath, "trigram_freq.txt"), 'w') as fw:
    #     for word in ngram_freq_list:
    #         for pos in word:
    #             fw.write("{}\t".format(pos))
    #         fw.write("\n")

    # np.save(os.path.join(featurePath, "trigram_freq.npy"), np.array(ngram_freq_list))

    # with open(os.path.join(featurePath, "bigram_proba_all.txt"), 'w') as fw:
    #     for freq in ngram_proba_list:
    #         fw.write("{}\n".format(freq))

    np.save(os.path.join(featurePath, "bigram_proba_all.npy"), np.array(ngram_proba_list))

def embeddingProcess(embeddingPath, featurePath):
    numlist = [100]
    windows = [2, 3, 4, 5, 6]
    for dimension in numlist:
        for w in windows:
            wordVectorPath = os.path.join(embeddingPath, '{}_{}_wordvec.txt'.format(dimension, w))
            wordvector_dict, vector_len = readVector(wordVectorPath)

            wordvector_list = []
            wordvector_len = dimension

            for word in wordlist:
                if not word in wordvector_dict.keys():
                    wordvector_list.append([0 for i in range(wordvector_len)])
                else:
                    wordvector_list.append(wordvector_dict[word])

            # with open(os.path.join(featurePath, "embedding_{}_{}.txt".format(dimension, w)), 'w') as fw:
            #     for word in wordvector_list:
            #         for pos in word:
            #             fw.write("{}\t".format(pos))
            #         fw.write("\n")

            np.save(os.path.join(featurePath, "embedding_{}_{}.npy".format(dimension, w)), np.array(wordvector_list))


def embeddingProcess_npy(embeddingPath, featurePath):
    dimensions = [300]
    windows = [5]
    for dimension in dimensions:
        for w in windows:
            wordVectorPath = os.path.join(embeddingPath, '{}_{}_wordvec.pkl'.format(dimension, w))
            with open(wordVectorPath, 'rb') as fr:
                wordvector_dict = pickle.load(fr)

            # wordvector_dict, vector_len = readVector(wordVectorPath)

            wordvector_list = []
            wordvector_len = dimension

            for word in wordlist:
                word = word
                if not word in wordvector_dict.keys():
                    wordvector_list.append([0 for i in range(wordvector_len)])
                else:
                    wordvector_list.append(wordvector_dict[word])

            # with open(os.path.join(featurePath, "embedding_{}_{}.txt".format(dimension, w)), 'w') as fw:
            #     for word in wordvector_list:
            #         for pos in word:
            #             fw.write("{}\t".format(pos))
            #         fw.write("\n")

            np.save(os.path.join(featurePath, "embedding_{}_{}.npy".format(dimension, w)), np.array(wordvector_list))
            print("dimension: {}, window: {} is finished.".format(dimension, w))

def typeProcess(typePath, featurePath):
    pos_binary_path = os.path.join(typePath, "type_vector.txt")
    pos_freq_path = os.path.join(typePath, "type_vector_freq.txt")


    pos_binary_dict, _ = readVector(pos_binary_path)
    pos_freq_dict = readVector(pos_freq_path)

    pos_binary_list, pos_freq_list = [], []

    # pos_binary_len = len(pos_binary_dict[b"meistens"])
    pos_binary_len = len(pos_binary_dict["画"])
    for word in wordlist:
        if not word in pos_binary_dict.keys():
            pos_binary_list.append([0 for i in range(pos_binary_len)])
        else:
            pos_binary_list.append(pos_binary_dict[word])

        # if not word in pos_freq_dict.keys():
        #     pos_freq_list.append([0 for i in range(pos_freq_len)])
        # else:
        #     pos_freq_list.append(pos_freq_dict[word])

    # with open(os.path.join(featurePath, "type_binary.txt"), 'w') as fw:
    #     for word in pos_binary_list:
    #         for pos in word:
    #             fw.write("{}\t".format(pos))
    #         fw.write("\n")

    np.save(os.path.join(featurePath, "type_binary.npy"), np.array(pos_binary_list))

    # with open(os.path.join(featurePath, "type_freq.txt"), 'w') as fw:
    #     for word in pos_freq_list:
    #         for pos in word:
    #             fw.write("{}\t".format(pos))
    #         fw.write("\n")
    #
    # np.save(os.path.join(featurePath, "type_freq.npy"), np.array(pos_freq_list))

def pronProcess(pronPath, featurePath):
    pron_binary_path = os.path.join(pronPath, "word_pronunciation.txt")

    pronunciations = []

    with open(pron_binary_path, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            pronunciations.append(line[1:])

    with open(os.path.join(featurePath, "pronunciation_binary.txt"), 'w') as fw:
        for word in pronunciations:
            for pos in word:
                fw.write("{}\t".format(pos))
            fw.write("\n")

    np.save(os.path.join(featurePath, "pronunciation_binary.npy"), np.array(pronunciations))

def LenProcess(featurePath):
    lenlist = []
    for word in wordlist:
        lenlist.append(len(word))

    np.save(os.path.join(featurePath, "len.npy"), np.array(lenlist))

def embeddingProcessDe(embeddingPath, featurePath):
    wordVectorPath = os.path.join(embeddingPath, "wordvec.txt")
    wordvector_dict, vector_len = readVector(wordVectorPath)

    wordvector_list = []
    wordvector_len = vector_len

    for word in wordlist:
        if not word in wordvector_dict.keys():
            wordvector_list.append([0 for i in range(wordvector_len)])
        else:
            wordvector_list.append(wordvector_dict[word])

    with open(os.path.join(featurePath, "embedding_{}.txt".format(vector_len)), 'w') as fw:
        for word in wordvector_list:
            for pos in word:
                fw.write("{}\t".format(pos))
            fw.write("\n")

    np.save(os.path.join(featurePath, "embedding_{}.npy".format(vector_len)), np.array(wordvector_list))

def glovePrpcessEn(embeddingPath, featurePath):
    for dimension in [150, 200, 250, 300]:
        glovePath = os.path.join(embeddingPath, 'vectors_{}_10.txt'.format(dimension))
        glove_dict = {}
        with open(glovePath, 'r') as fr:
            for line in fr:
                linelist = line.strip().split()
                word, wordvec = linelist[0], linelist[1:]

                glove_dict[word] = wordvec

        wordvector_list = []
        wordvector_len = dimension

        for word in wordlist:
            word = word.decode()
            if not word in glove_dict.keys():
                wordvector_list.append([0 for i in range(wordvector_len)])
            else:
                wordvector_list.append(glove_dict[word])

        np.save(os.path.join(featurePath, "glove_{}_10.npy".format(dimension)), np.array(wordvector_list))
        print("dimension: {} is finished.".format(dimension))

def glovePrpcessDe(embeddingPath, featurePath):
    glovePath = os.path.join(embeddingPath, 'vectors.txt')
    glove_dict = {}
    wordvector_len = 0
    with open(glovePath, 'r') as fr:
        for line in fr:
            linelist = line.strip().split()
            wordvector_len = len(linelist) - 1
            word, wordvec = linelist[0], linelist[1:]

            glove_dict[word] = wordvec

    wordvector_list = []

    for word in wordlist:
        word = word.decode()
        if not word in glove_dict.keys():
            wordvector_list.append([0 for i in range(wordvector_len)])
        else:
            wordvector_list.append(glove_dict[word])

    np.save(os.path.join(featurePath, "300_10.npy"), np.array(wordvector_list, dtype=np.float64))
    print("German is finished.")

def glovePrpcessCN(embeddingPath, featurePath):
    glovePath = os.path.join(embeddingPath, 'cn_300_10.txt')
    glove_dict = {}
    wordvector_len = 0
    with open(glovePath, 'r') as fr:
        for line in fr:
            linelist = line.strip().split()
            wordvector_len = len(linelist) - 1
            word, wordvec = linelist[0], linelist[1:]

            glove_dict[word] = wordvec

    wordvector_list = []

    for word in wordlist:
        if not word in glove_dict.keys():
            wordvector_list.append([0 for i in range(wordvector_len)])
        else:
            wordvector_list.append(glove_dict[word])

    np.save(os.path.join(featurePath, "300_10.npy"), np.array(wordvector_list, dtype=np.float64))
    print("German is finished.")


if __name__ == '__main__':
    # corpus = "nytimes"
    # corpus = "gutenberg"
    # corpus = "E1E2"
    corpus = "chinese"
    feature_path = "/home/sophie/WordsLevel/features/{}/single_feature".format(corpus)
    # feature_path = "/home/sophie/WordsLevel/features/{}/rebuttal_feature".format(corpus)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    # wordlist = getWordList("/home/sophie/WordsLevel/wordlist.txt")
    wordlist = getWordList("/home/sophie/WordsLevel/word_all_cn.txt")

    # Frequency
    # freqPath = "/home/sophie/WordsLevel/Freq/nytimes/freq_sort_from_lem.txt"
    # freqPath = "/home/sophie/WordsLevel/Freq/chinese/freq_sort_filter.txt"
    # FreqProcess(freqPath, feature_path)

    # Len
    # LenProcess(feature_path)

    # Pos
    # posPath = "/home/sophie/WordsLevel/pos/chinese/"
    # posProcess(posPath, feature_path)

    # uPos
    posPath = "/home/sophie/WordsLevel/pos/chinese/"
    posProcess(posPath, feature_path)

    # bigram & trigram
    # ngramPath = "/home/sophie/WordsLevel/bigram/chinese"
    # bigramAndTrigram(ngramPath, feature_path)

    # word embedding
    # embeddingPath = "/home/sophie/WordsLevel/word2vec/chinese"
    # # embeddingProcess(embeddingPath, feature_path)
    # embeddingProcess_npy(embeddingPath, feature_path)

    # Glove
    # GlovePath = "/home/sophie/WordsLevel/glove/{}".format(corpus)
    # # glovePrpcessEn(GlovePath, feature_path)
    # glovePrpcessCN(GlovePath, feature_path)

    # type dependency
    # dependencyPath = "/home/sophie/WordsLevel/parsing/chinese"
    # typeProcess(dependencyPath, feature_path)

    # pronunciation
    # pronunciationPath = "/home/sophie/WordsLevel/pron/chinese/"
    # pronProcess(pronunciationPath, feature_path)