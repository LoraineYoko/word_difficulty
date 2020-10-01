# -*- coding: utf-8 -*-
# @Time : 2019/4/13 下午9:23
# @Author : Sophie_Zhang
# @File : conParsing.py
# @Software: PyCharm

import os
from collections import defaultdict

def conIndexFilter():
    type_list = set()
    with open("typed_dependencies_manual.txt", 'r') as f:
        for line in f:
            type_list.add("{}_in".format(line.strip()))
            type_list.add("{}_out".format(line.strip()))

    with open("conParsing_filter.txt", "w") as fw:
        with open("conParsing.txt", 'r') as fr:
            for line in fr:
                try:
                    __type = line.strip().split("\t")[1]
                    if __type in type_list:
                        fw.write(line)
                except:
                    pass

def ListReplace(l, newItem, place):
    origin_freq = l[place]
    del l[place]
    l.insert(place, origin_freq + newItem)
    return l

def ListReplaceBinary(l, place):
    del l[place]
    l.insert(place, 1)
    return l

def conCount():
    path = "conParsing_filter.txt"
    word_pos_dict = {}
    pos_dict = defaultdict(int)
    with open(path, "r") as fr:
        cnt = 0
        for line in fr:
            line = line.strip().split("\t")
            word = line[0]
            pos = line[1]
            if word in word_pos_dict.keys():
                # posdict = word_pos_dict[word]
                # if pos in posdict:
                #     posdict[pos] += 1
                # else:
                #     posdict[pos] = 1
                # word_pos_dict[word] = posdict
                posdict = word_pos_dict[word]
                posdict[pos] += 1
                word_pos_dict[word] = posdict
            else:
                # posdict = {}
                # posdict[pos] = 1
                # word_pos_dict[word] = posdict
                posdict = defaultdict(int)
                posdict[pos] += 1
                word_pos_dict[word] = posdict
            # cnt += 1
            # if cnt == 100:
            #     break

    with open("/home/sophie/WordsLevel/parsing/E1E2/word_con_filter.txt", "w") as f:
        for word in word_pos_dict.keys():
            posdict = word_pos_dict[word]
            f.write(word + "\t")
            for pos in posdict.keys():
                f.write("{}:{}\t".format(pos, posdict[pos]))
            f.write("\n")

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
    path = "data/"
    vector_freq_path = os.path.join(path, "type_vector_freq.txt")
    vector_path = os.path.join(path, "type_vector.txt")

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
    # conIndexFilter()
    # conCount()
    wordpath = "wordlist.txt"
    wordlist = []
    with open(wordpath, 'r') as fr:
        for line in fr:
            wordlist.append(line.strip())

    pos_path = "data/word_con_filter.txt"
    # writeBooksFreq()
    POSList = getPosList("data/conParsing_filter.txt")
    print(len(list(POSList)))
    POSVectors(pos_path, wordlist)