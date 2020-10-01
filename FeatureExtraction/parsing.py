# -*- coding: utf-8 -*-
# @Time : 2019/4/17 上午11:05
# @Author : Sophie_Zhang
# @File : gutenberg_np.py
# @Software: PyCharm

from pycorenlp import StanfordCoreNLP
import time
import datetime
import os
import json
import sys

'''
nohup java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000 > pipeline.log 2>&1 &
nohup python -u gutenberg_np.py file_index 9000 > gutenberg_np.log 2>&1 &
'''

def docProcess(nlp, path, parsingDir):
    cnt = 0
    with open(path, 'r') as fr, \
            open(os.path.join(parsingDir, "lemma.txt"), 'w') as lemf, \
            open(os.path.join(parsingDir, "deparsing.txt"), 'w') as parsef, \
            open(os.path.join(parsingDir, "conParsing.txt"), 'w') as denf:
        for line in fr:
            try:
                cnt += 1
                line = line.strip()

                lemma_sent_string = ""
                parsing_sent = ""
                dependencies_content = ""

                annotations = get_stanford_annotations(nlp, line, port=9000,
                                                       annotators='tokenize,ssplit,pos,lemma,depparse,parse')
                annotations = json.loads(annotations, encoding="utf-8", strict=False)

                tokens = annotations['sentences'][0]['tokens']
                token_list = [token['originalText'].lower() for token in tokens]
                index_list = [token['index'] for token in tokens]
                lemma_list = [token['lemma'].lower() for token in tokens]
                pos_list = [token['pos'] for token in tokens]

                for i, word in enumerate(lemma_list):
                    lemma_sent_string += word + " "
                    parsing_sent += "{} {} {}\n".format(word, index_list[i], pos_list[i])

                for edge in annotations['sentences'][0]['basicDependencies']:
                    dependency = "({},{},{})".format(lemma_list[edge['governor'] - 1], lemma_list[edge['dependent'] - 1], edge['dep'])
                    dependencies_content += dependency + "\n"

                # with open(os.path.join(lempath, filename), 'a') as f:
                lemf.write(lemma_sent_string + "\n")

                # with open(os.path.join(parsingpath, filename), 'a') as f:
                parsef.write(parsing_sent + "\n")

                # with open(os.path.join(dependencypath, filename), 'a') as f:
                denf.write(dependencies_content)

                print("{}: \t{} line is finished.".format(datetime.datetime.now(), cnt))
            except:
                pass

def get_stanford_annotations(nlp, text, port=9000,
                             annotators='tokenize,ssplit,pos,lemma,depparse,parse'):
    output = nlp.annotate(text, properties={
        "timeout": "10000",
        "ssplit.isOneSentence": "true",
        'annotators': annotators,
    })
    return output

if __name__ == '__main__':
    # dirpath = sys.argv[1]
    dirpath = ""
    # port = sys.argv[2]
    port = 9000
    path = '/home/sophie/Document/DataProcess/gutenberg/gutenberg_ssplit_{}.txt'.format(dirpath)

    parsingDir = '/home/sophie/WordsLevel/corpus/gutenberg/parsing-docs/{}/'.format(dirpath)
    if not os.path.exists(parsingDir):
        os.makedirs(parsingDir)

    nlp = StanfordCoreNLP('http://localhost:{0}'.format(port))

    docProcess(nlp, path, parsingDir)