# Automatic Classification and Comparison of Words by Difficulty
The paper Automatic Classification and Comparison of Words by Difficulty has accepted by the 27th International Conference on Neural Information Processing (ICONIP 2020).

In this repository, you can find all the resource we used and the main codes. 

## Dataset
A dataset is made up of three parts: a reliable corpus (Corpus), a pronunciation dictionary (Pdict) and a standard leveled word list (W). 
Corpus and Pdict are the resource for extracting features which are stated in Sec. 2, W is regarded as the ground truth.

The following tables show the resources and details of the Corpus and W.
<table>
    <tr align="center">
        <td>Language</td>
        <td>Corpus</td>
    </tr>
    <tr align="center">
        <td rowspan="2">English</td>
        <td><a href="https://drive.google.com/file/d/1IPF3IzNDVASysctL3sjKB6YLidxIHV5m/view?usp=sharing">NewYork Times (2005-2006)</a></td>
    </tr>
    <tr align="center">
        <td><a href="https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html">Gutenberg</a></td>
    </tr>
    <tr align="center">
        <td>German</td>
        <td><a href="https://drive.google.com/file/d/1mNgEyKJJtILiTspBZWwJcFcxWOoLJavG/view?usp=sharing">Parallel Corpus for German</a></td>
    </tr>
    <tr align="center">
        <td>Chinese</td>
        <td><a href="https://dumps.wikimedia.org/zhwiki/latest/">Wikipedia for Chinese</a></td>
    </tr>
</table>

The details of our ground truth are listed as following:
![](https://github.com/LoraineYoko/word_difficulty/blob/master/figure/GT.png)

## Results
Due to the limitation of the paper, all the experimental results are shown here.

This table shows the classification and ranking results using two English Corpus and their combination to extract the features. 
The accuracy for baseline models and our feature engineering models is the average of ten runs. 
Test means the accuracy on test set and CV means the accuracy on cross validation. 
MFF is the multi-faceted features using Word2Vec to obtain word embeddings. 
(** indicates p-value ≤ 0.01 compared with Random, FC, FO, FPOS and FLSCP baselines.)

![](https://github.com/LoraineYoko/word_difficulty/blob/master/figure/EN.png?raw=true)

This table shows the classification and ranking results using German and Chinese Corpus to extract features. (** indicates p-value ≤ 0.01 compared with Random, FC, FO, FPOS and FLSCP baselines; †† indicates p-value ≤ 0.01 compared with Random, FC, FO and FPOS baselines; ‡‡ indicates p-value ≤ 0.01 compared with Random and FC baselines.)
![](https://github.com/LoraineYoko/word_difficulty/blob/master/figure/GE%20and%20CN.png)
