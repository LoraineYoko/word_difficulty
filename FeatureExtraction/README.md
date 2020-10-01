# Codes for feature extraction
* length: Easy to obtain.
* pronounciation: "from nltk.corpus import cmudict"
* bigram: Bigram.py
* trigram: Trigram.py
* frequency & pos tag & parsing: 
    * use parsing.py to do parsing 
    * use POS.py to ontain POS vector
    * use conParsing.py to ontain dependency type vector
    * use lemma txt to count frequency
* word embedding: Word2Vec.py

* feature combination:
    * preprocess.py: convert each feature file to npy format
    * featureExtraction.py: feature combination, including all features and features for ablation tests
