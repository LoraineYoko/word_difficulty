# -*- coding: utf-8 -*-
# @Time : 2019/4/12 下午9:49
# @Author : Sophie_Zhang
# @File : POS.py
# @Software: PyCharm

import os
import datetime
from math import log
import numpy as np

def ListReplace(l, newItem, place):
    origin_freq = l[place]
    del l[place]
    l.insert(place, origin_freq + newItem)
    return l

def ListReplaceBinary(l, place):
    del l[place]
    l.insert(place, 1)
    return l

def getwordPOSDict(posPath):
    wordPOSDict = {}
    wordList =[]
    with open(posPath, 'r') as f:
        for line in f:
            try:
                linelist = line.strip().split('\t')
                word = linelist[0]
                wordList.append(word)
                pos_dict = {}
                for pos in linelist[1:]:
                    poslist = pos.split(':')
                    pos_dict[poslist[0]] = int(poslist[1])
                wordPOSDict[word] = pos_dict
            except:
                print("Line error.")
    return wordPOSDict, wordList

def POSVecGeneration(word, wordPOSDict, POSList, length):
    word_pos_list = [0 for i in range(length)]
    if word in wordPOSDict.keys():
        word_pos_dict = wordPOSDict[word]
        for pos in word_pos_dict.keys():
            if pos in POSList:
                appearenceTime = word_pos_dict[pos]
                pos_index = POSList.index(pos)
                # print("pos index:", pos_index)
                word_pos_list = ListReplace(word_pos_list, appearenceTime, pos_index)
    # print(word_pos_list)
    # print(len(word_pos_list))
    return word_pos_list

def POSVecGenerationBinary(word, wordPOSDict, POSList, length):
    word_pos_list = [0 for i in range(length)]
    if word in wordPOSDict.keys():
        word_pos_dict = wordPOSDict[word]
        for pos in word_pos_dict.keys():
            if pos in POSList:
                pos_index = POSList.index(pos)
                # print("pos index:", pos_index)
                word_pos_list = ListReplaceBinary(word_pos_list, pos_index)

    # print(word_pos_list)
    # print(len(word_pos_list))
    return word_pos_list


def POSVectors(posPath, wordlist):
    '''
        pos: ['EX', 'NNP', 'VBD', 'DT', 'RB', 'JJ', 'NN', 'IN',
         'VBN', 'PRP', 'CC', 'VBZ', 'TO', 'VB', 'VBG',
          'VBP', 'NNS', 'CD', 'RP', 'PRP$', 'MD', 'JJR', 'PDT',
           'FW', 'WDT', 'WP', 'UH', 'WRB', 'RBR', 'JJS',
            'NNPS', 'RBS', 'POS', 'SYM', 'WP$', 'LS']
    '''

    path = "/home/sophie/WordsLevel/pos/E1E2/"
    vector_freq_path = os.path.join(path, "xpos_vector_freq.txt")
    vector_path = os.path.join(path, "xpos_vector.txt")

    # POSList = ['EX', 'NNP', 'VBD', 'DT', 'RB', 'JJ', 'NN', 'IN',
    #            'VBN', 'PRP', 'CC', 'VBZ', 'TO', 'VB', 'VBG', 'VBP',
    #            'NNS', 'CD', 'RP', 'PRP$', 'MD', 'JJR', 'PDT', 'FW',
    #            'WDT', 'WP', 'UH', 'WRB', 'RBR', 'JJS', 'NNPS', 'RBS',
    #            'POS', 'SYM', 'WP$', 'LS']

    # POSList = ['SYM', 'ADD', 'DET', 'ADJ', 'PROPN', 'PUNCT', 'ADP', 'NOUN',
    #            'ADV', 'PRON', 'X', 'AUX', 'PART', 'NNP', 'SCONJ', 'NUM', 'CCONJ', 'INTJ', 'VERB']

    wordPOSDict, _ = getwordPOSDict(posPath)

    with open(vector_freq_path, 'w') as fw:
        for word in wordlist:
            word_pos_list = POSVecGeneration(word, wordPOSDict, POSList, len(POSList))
            # word_pos_list = POSVecGenerationBinary(word, wordPOSDict, POSList)
            fw.write(word+"\t")
            for pos in word_pos_list:
                fw.write(str(pos)+"\t")
            fw.write('\n')

    with open(vector_path, 'w') as fw:
        for word in wordlist:
            # word_pos_list = POSVecGenerationBinary(word, wordPOSDict, POSList, len(POSList))
            word_pos_list = POSVecGenerationBinary(word, wordPOSDict, POSList, len(POSList))
            fw.write(word+"\t")
            for pos in word_pos_list:
                fw.write(str(pos)+"\t")
            fw.write('\n')

def getPosList(path):
    poset = set()
    with open(path, 'r') as f:
        for line in f:
            pos = line.strip().split("\t")[1]
            poset.add(pos)
    return list(poset)

if __name__ == '__main__':
    wordpath = ""
    wordlist = []
    with open(wordpath, 'r') as fr:
        for line in fr:
            wordlist.append(line.strip())

    pos_path = "xpos.txt"
    # writeBooksFreq()
    POSList = getPosList("all_pos.txt")
    print(len(list(POSList)))

    # POSList = getPosList("/home/sophie/WordsLevel/corpus/nytimes/nytimes-corpus/parsing-docs/xpos.txt")
    # print(len(list(POSList)))
    POSVectors(pos_path, wordlist)
    # POSNormalization()
    # POSNormalization2ZeroOne()