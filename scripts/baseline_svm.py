# re-implement SVM baseline in Barbieri et al., 2020
# Author: Jinghua Xu

import random
import re
import warnings
import numpy as np
import pandas as pd
from numpy import array
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
# from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
# from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from src.src import load_tweets, load_labels, preprocess

# encoding class


class Encoding:
    def __init__(self, gold_tok):
        self.gold_tok = gold_tok

    def word_ngrams(self, paragraph, min=1, max=3):
        word_vect = TfidfVectorizer(
            analyzer='word', token_pattern=r'\w{1,}', ngram_range=(min, max))
        word_vect.fit(self.gold_tok)
        word_ngrams = word_vect.transform(paragraph)
        return word_ngrams

    def char_ngrams(self, paragraph, min=1, max=5):
        char_vector = TfidfVectorizer(analyzer='char', ngram_range=(min, max))
        char_vector.fit(self.gold_tok)
        char_ngrams = char_vector.transform(paragraph)
        return char_ngrams


if __name__ == '__main__':

    # file paths
    train_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/train_text.txt'
    train_labels_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/train_labels.txt'

    val_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/val_text.txt'
    val_labels_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/val_labels.txt'

    test_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_text.txt'
    test_labels_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_labels.txt'

    # load data
    train_labels = load_labels(train_labels_path)
    train_txt = load_tweets(train_txt_path)

    val_labels = load_labels(val_labels_path)
    val_txt = load_tweets(val_txt_path)

    test_labels = load_labels(test_labels_path)
    test_txt = load_tweets(test_txt_path)

    # WORD & CHAR N-GRAMS FEATURES
    feats = Encoding(train_txt)
    word_ngrams_train = feats.word_ngrams(train_txt)
    word_ngrams_val = feats.word_ngrams(val_txt)
    word_ngrams_test = feats.word_ngrams(test_txt)
    char_ngrams_train = feats.char_ngrams(train_txt)
    char_ngrams_val = feats.char_ngrams(val_txt)
    char_ngrams_test = feats.char_ngrams(test_txt)

    # stack features
    feats_train = hstack([word_ngrams_train, char_ngrams_train])
    feats_val = hstack([word_ngrams_val, char_ngrams_val])
    feats_test = hstack([word_ngrams_test, char_ngrams_test])

    # # search hyperparameters
    # param_grid = {'C': [0.1, 1, 10, 100], 'dual': [True, False], 'loss': [
    #     'hinge', 'squared_hinge'], 'penalty': ['l1', 'l2'], 'max_iter': [100, 500, 1000, 1500, 2000], 'class_weight': ['balanced']}
    # grid = GridSearchCV(LinearSVC(), param_grid,
    #                     refit=True, verbose=2, scoring='f1')

    model = LinearSVC(C=1, class_weight='balanced', dual=False, max_iter=500, penalty='l1')
    model.fit(feats_train, train_labels)
    # print(grid.best_estimator_)
    grid_predictions = model.predict(feats_test)

    preds_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds3.txt'
    # parameters_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/svm_tuned_params.txt'

    with open(preds_path, 'w') as f:
        for grid_prediction in grid_predictions:
            f.write(str(grid_prediction)+'\n')

    # with open(parameters_path, 'w') as f:
    #     f.write(grid.best_estimator_)

    # print(f1_score(y_test, grid_predictions))
    # print(precision_score(y_test, grid_predictions))
    # print(recall_score(y_test, grid_predictions))