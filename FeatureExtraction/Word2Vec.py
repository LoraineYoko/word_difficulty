# -*- coding: utf-8 -*-
# @Time : 2019/3/20 上午10:32
# @Author : Sophie_Zhang
# @File : Word2Vec.py
# @Software: PyCharm

import os
import datetime
import pickle
from gensim.models import word2vec
from multiprocessing import Pool
import numpy as np

def WriteVec2Txt(desPath, word, WordVector):
    with open(desPath, 'a+', encoding='utf-8') as f:
        f.write(word + "\t")
        WordVectorList = WordVector.tolist()
        for num in WordVectorList:
            f.write(str(num)+"\t")
        f.write("\n")

def Word2VectorModelGeneration(model_path, file, dimension, w):
    print("Getting the word2vec model...dimension:{}, window:{}".format(dimension, w))
    sentences = word2vec.Text8Corpus(file)
    # sentences = word2vec.Text8Corpus("test")
    # sg=1 skip-gram算法
    model = word2vec.Word2Vec(sentences, sg=1, size=dimension, window=w, min_count=1, workers=10, trim_rule=None)
    print("Model is ok.")
    model.save(os.path.join(model_path, '{}-{}.model'.format(dimension, w)))
    return model

def getWord2Vec(model_path, file_path, d, w, corpus, desdir):
    model = Word2VectorModelGeneration(model_path, file_path, d, w)
    vocabularyList = list(model.wv.vocab)
    wordVec_dict = {}
    for word in vocabularyList:
        try:
            wordVector = model[word]
            # WriteVec2Txt(os.path.join(desdir, '{}_{}_wordvec.txt'.format(d, w)), word, wordVector)
            # print("word '{}' finished.".format(word))
            wordVec_dict[word] = wordVector
        except:
            print("No such word '{}' in model".format(word))

    pickle.dump(wordVec_dict,
                os.path.join(desdir, '{}_{}_wordvec.pkl'.format(d, w)))

    print("{}: corpus {} dimension {} window {} finished.".
          format(datetime.datetime.now(),
                 corpus, d, w))

def save_word2vec(model_path, d, w, corpus, desdir):
    model = word2vec.Word2Vec.load(os.path.join(model_path, '{}-{}.model'.format(d, w)))

    # vocab = np.load("word2vec_model/gutenberg/100-2.model.wv.vectors.npy")
    # print(vocab)
    vocabularyList = list(model.wv.vocab)
    # print(vocabularyList)

    wordVec_dict = {}
    for word in vocabularyList:
        try:
            wordVector = model[word]
            wordVec_dict[word] = wordVector
        except:
            print("No such word '{}' in model".format(word))

    with open(os.path.join(desdir, '{}_{}_wordvec.pkl'.format(d, w)), 'wb') as fr:
        pickle.dump(wordVec_dict, fr)

    print("{}: corpus {} dimension {} window {} finished.".
          format(datetime.datetime.now(),
                 corpus, d, w))

def main(model_path, file_path, corpus):
    desdir = '/home/sophie/WordsLevel/word2vec/{}/'.format(corpus)
    if not os.path.exists(desdir):
        os.makedirs(desdir)
    dimensions = [100, 150, 200, 250, 300]
    windows = [2, 3, 4, 5, 6]

    # pool = Pool(processes=5)

    for d in dimensions:
        for w in windows:
            try:
                # model = Word2VectorModelGeneration(model_path, file_path, d, w)
                save_word2vec(model_path, d, w, corpus, desdir)
            except:
                print("{}: corpus {} dimension {} window {} error.".
                      format(datetime.datetime.now(), corpus, d, w))
            # pool.apply_async(getWord2Vec, (model_path, file_path, d, w, corpus, desdir))
            # pool.apply_async(save_word2vec, (model_path, d, w, corpus, desdir))
    # pool.close()
    # pool.join()

if __name__ == '__main__':
    path1 = ""
    path2 = ""
    path3 = ""
    files = [path1, path2, path3]
    corpora = ["gutenberg", "nytimes", "E1E2"]


    for i in [0, 1]:
        model_path = "word2vec_model/{}/".format(corpora[i])
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        main(model_path, files[i], corpora[i])


