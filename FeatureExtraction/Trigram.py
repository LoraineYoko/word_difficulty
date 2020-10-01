# -*- coding: utf-8 -*-
# @Time : 2019/4/12 下午9:29
# @Author : Sophie_Zhang
# @File : Trigram.py
# @Software: PyCharm

import os
import sys
import nltk
from collections import defaultdict
from math import log

class Trigram(object):
    def __init__(self, wordlist):
        self.wordlist = wordlist
        self.TrigramTruple, self.TrigramList = self.getAllWordTrigram(wordlist=wordlist)
        self.TrigramProb = self.getTrigramFreq()
        self.DictionaryProb = self.getDictionaryProb()

    def getWordTrigram(self, word):
        letters = [l for l in word]
        if len(letters) == 1:
            trigram = [(b'$s', word, b"$s")]
        elif len(letters) == 2:
            trigram = [(b'$s', word), (word, b'$s')]
        else:
            trigram = list(nltk.trigrams(letters))
            begin_tuple, end_tuple = (b'$s', letters[0], letters[1]), \
                                     (letters[-2], letters[-1], b'$s')

            trigram.append(begin_tuple)
            trigram.append(end_tuple)
        # print(word, trigram)
        return trigram

    def getAllWordTrigram(self, wordlist):
        wordsTrigram = []
        for word in wordlist:
            wordsTrigram.extend(self.getWordTrigram(word))
        return (tri for tri in wordsTrigram), wordsTrigram

    def getTrigramFreq(self):
        return nltk.FreqDist(self.TrigramList)

    def getDictionaryProb(self):
        DictionaryProbDict = {}

        # DictionaryDist = nltk.LaplaceProbDist(self.TrigramProb)
        DictionaryDist = nltk.KneserNeyProbDist(self.TrigramProb)
        for i in DictionaryDist.samples():
            DictionaryProbDict[i] = DictionaryDist.prob(i)

        return DictionaryProbDict

    def getWordProb(self, word):
        wordTigram = self.getWordTrigram(word)
        wordProb = 0.0
        for tri in wordTigram:
            tri_prob = self.DictionaryProb[tri]
            # tri_prob = self.
            # wordProb += tri_prob
            wordProb += log(tri_prob, 10)
        return wordProb

    def getAllWordsProb(self):
        wordsProbDict = {}
        for word in self.wordlist:
            wordProb = self.getWordProb(word)
            wordsProbDict[word] = wordProb
        return wordsProbDict

def ListReplaceBinary(l, place):
    del l[place]
    l.insert(place, 1)
    return l

def TrigramGenerationBinary(trigram, word, TrigramList, length):
    wordTrinarm = trigram.getWordTrigram(word)
    word_trigram_list = [0 for i in range(length)]
    for bi in wordTrinarm:
        if bi in TrigramList:
            pos_index = TrigramList.index(bi)
            # print("pos index:", pos_index)
            word_trigram_list = ListReplaceBinary(word_trigram_list, pos_index)
    # print(word, ":", word_bigram_list)
    return word_trigram_list

def ListReplace(l, place):
    Freq = l[place] + 1
    del l[place]
    l.insert(place, Freq)
    return l

def TrigramGeneration(trigram, word, TrigramList, length):
    wordTrinarm = trigram.getWordTrigram(word)
    word_trigram_list = [0 for i in range(length)]
    for bi in wordTrinarm:
        if bi in TrigramList:
            pos_index = TrigramList.index(bi)
            # print("pos index:", pos_index)
            word_trigram_list = ListReplace(word_trigram_list, pos_index)
    # print(word, ":", word_bigram_list)
    return word_trigram_list

def TrigramProcess(wordlist, path):

    vector_freq_path = os.path.join(path, "vector_freq.txt")
    vector_path = os.path.join(path, "vector.txt")


    trigram = Trigram(wordlist)

    trigramItems = list(set(trigram.TrigramList))
    print(len(trigramItems))

    with open(vector_freq_path, 'wb') as f:
        for word in wordlist:
            word_trigram_list = TrigramGeneration(trigram, word, trigramItems, len(trigramItems))
            f.write(word + b'\t')
            for tri in word_trigram_list:
                f.write(str(tri).encode() + b"\t")
            f.write(b'\n')

    with open(vector_path, 'wb') as f:
        for word in wordlist:
            word_trigram_list = TrigramGenerationBinary(trigram, word, trigramItems, len(trigramItems))
            f.write(word + b'\t')
            for tri in word_trigram_list:
                f.write(str(tri).encode() + b"\t")
            f.write(b'\n')

def checktrigram(allwordlist):
    trigram = Trigram(allwordlist)
    TrigramList = trigram.TrigramList
    trigramDict = defaultdict(int)
    for tri in TrigramList:
        trigramDict[tri] += 1

    trilist = sorted(trigramDict.items(), key=lambda x: x[1], reverse=True)
    print(trilist[:100])

def probaGen(wordlist, allwordlist, path):
    if not os.path.exists(path):
        os.makedirs(path)
    proba_path = os.path.join(path, "probability_all_log.txt")

    trigram = Trigram(allwordlist)
    wordsProbDict = trigram.getAllWordsProb()
    with open(proba_path, 'wb') as f:
        for word in allwordlist:
            proba = wordsProbDict[word]
            f.write(("{}\t{}\n".format(word.decode(), proba)).encode())


if __name__ == '__main__':
    # Words of groug truth
    wordpath = sys.argv[1]
    # Words of corpus
    allwordpath = sys.argv[2]
    # To save the LM proba of words
    proba_des_path = sys.argv[3]


    wordlist = []
    with open(wordpath, 'rb') as fr:
        for line in fr:
            if not b"-" in line:
                wordlist.append(line.strip())

    websterlist = set()
    with open(websterpath, 'rb') as fr:
        for line in fr:
            websterlist.add(line.strip().lower())

    allwordlist = []
    with open(allwordpath, 'rb') as fr:
        for line in fr:
            word = line.strip().split()[0]
            # if word in websterlist:
            allwordlist.append(word)

    probaGen(wordlist, allwordlist, proba_des_path)
    TrigramProcess(wordlist, proba_des_path)
    # checktrigram(allwordlist)
