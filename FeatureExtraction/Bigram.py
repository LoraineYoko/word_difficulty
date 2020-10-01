# -*- coding: utf-8 -*-
# @Time : 2019/4/12 下午6:32
# @Author : Sophie_Zhang
# @File : Bigram.py
# @Software: PyCharm

import os
import sys
import nltk
from math import log

class Bigram(object):
    def __init__(self, wordlist, freqAlgorithm):
        self.wordlist = wordlist
        self.freqAlgorithm = freqAlgorithm
        self.wordsBigramTriple, self.wordsBigramList = self.getAllWordBigram(self.wordlist)
        self.BigramFreqs = self.getBigramFreq()
        self.LaplaceProb = self.getLaplaceProb()
        self.DictionaryProb = self.getDictionaryProb()


    def getOneWordBigram(self, word):
        letters = [l for l in word]
        if len(letters) == 1:
            bigram = [(b'$s', word), (word, b'$s')]
        else:
            bigram = list(nltk.bigrams(letters))
            begin_tuple, end_tuple = (b'$s', str(bigram[0][0])), (str(bigram[-1][-1]), b'$s')

            bigram.append(begin_tuple)
            bigram.append(end_tuple)
        return bigram

    def getAllWordBigram(self, wordlist):
        wordsBigram = []
        for word in wordlist:
            wordsBigram.extend(self.getOneWordBigram(word))
        return (bi for bi in wordsBigram), wordsBigram

    def getBigramFreq(self):
        return nltk.FreqDist(self.wordsBigramList)

    def getLaplaceProb(self):
        LaplaceProbDict = {}
        LaplaceDist = nltk.LaplaceProbDist(self.BigramFreqs)
        for i in LaplaceDist.samples():
            LaplaceProbDict[i] = LaplaceDist.prob(i)

        return LaplaceProbDict

    def getDictionaryProb(self):
        DictionaryProbDict = {}
        DictionaryDist = nltk.LaplaceProbDist(self.BigramFreqs)
        # DictionaryDist = nltk.KneserNeyProbDist(self.BigramFreqs)

        for i in DictionaryDist.samples():
            DictionaryProbDict[i] = DictionaryDist.prob(i)

        return DictionaryProbDict

    def getWordProb(self, word):
        wordBigram = self.getOneWordBigram(word)
        wordProb = 0.0
        for bi in wordBigram:
            if self.freqAlgorithm == "dict":
                bi_prob = self.DictionaryProb[bi]
                # wordProb += bi_prob
                wordProb += log(bi_prob, 10)
            elif self.freqAlgorithm == "LaplaceProb":
                bi_prob = self.LaplaceProb[bi]
                # wordProb += bi_prob
                wordProb += log(bi_prob, 10)
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

def BigramGenerationBinary(bigram, word, BigramList, length):
    wordBinarm = bigram.getOneWordBigram(word)
    word_bigram_list = [0 for i in range(length)]
    for bi in wordBinarm:
        if bi in BigramList:
            pos_index = BigramList.index(bi)
            # print("pos index:", pos_index)
            word_bigram_list = ListReplaceBinary(word_bigram_list, pos_index)
    # print(word, ":", word_bigram_list)
    return word_bigram_list

def ListReplace(l, place):
    freq = l[place] + 1
    del l[place]
    l.insert(place, freq)
    return l

def BigramGeneration(bigram, word, BigramList, length):
    wordBinarm = bigram.getOneWordBigram(word)
    word_bigram_list = [0 for i in range(length)]
    for bi in wordBinarm:
        if bi in BigramList:
            pos_index = BigramList.index(bi)
            # print("pos index:", pos_index)
            word_bigram_list = ListReplace(word_bigram_list, pos_index)
    # print(word, ":", word_bigram_list)
    return word_bigram_list

def BinaryProcess(wordlist, path):
    vector_freq_path = os.path.join(path, "vector_freq.txt")
    vector_path = os.path.join(path, "vector.txt")
    bigram = Bigram(wordlist, "dict")

    bigramItems = list(set(bigram.wordsBigramList))
    print(len(bigramItems))

    with open(vector_freq_path, 'wb') as f:
        for word in wordlist:
            # word_bigram_list = BigramGenerationBinary(bigram, word, bigramItems, len(bigramItems))
            word_bigram_list = BigramGeneration(bigram, word, bigramItems, len(bigramItems))
            f.write(word+b'\t')
            for bi in word_bigram_list:
                f.write(str(bi).encode() + b"\t")
            f.write(b'\n')

    with open(vector_path, 'wb') as f:
        for word in wordlist:
            # word_bigram_list = BigramGenerationBinary(bigram, word, bigramItems, len(bigramItems))
            word_bigram_list = BigramGenerationBinary(bigram, word, bigramItems, len(bigramItems))
            f.write(word+b'\t')
            for bi in word_bigram_list:
                f.write(str(bi).encode() + b"\t")
            f.write(b'\n')

def probaGen(wordlist, allwordlist, path):
    if not os.path.exists(path):
        os.makedirs(path)
    proba_path = os.path.join(path, "probability_all_log.txt")

    bigram = Bigram(allwordlist, "dict")
    wordsProbDict = bigram.getAllWordsProb()
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
            wordlist.append(line.strip())

    allwordlist = []
    with open(allwordpath, 'rb') as fr:
        for line in fr:
            word = line.strip().split()[0]
            allwordlist.append(word)

    probaGen(wordlist, allwordlist, proba_des_path)
    BinaryProcess(wordlist)


