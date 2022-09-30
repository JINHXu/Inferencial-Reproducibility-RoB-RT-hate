# cerate dataset for reliability test of sota model
# Jinghua Xu

import pandas as pd
from collections import Counter
import string
import textstat
import math
# from readability import Readability
from wordfreq import word_frequency
from src.src import load_tweets, load_labels, preprocess
import os
import numpy as np

# concat all opts
input_dir = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/input_reliability_test/'
filepaths = [input_dir+f for f in os.listdir(
    input_dir) if f.endswith('.csv')]
df = pd.concat(map(pd.read_csv, filepaths))

test_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_text.txt'
tweets = load_tweets(test_txt_path)

# tweet length
tweet_len = []

# Flesch-Kincaid readability
readability = []

# word frequency
frequency = []

for tweet in tweets:
    tweet_len.append(len(tweet.split()))

    # Flesch-Kincaid readability
    readability.append(textstat.flesch_reading_ease(tweet))

    # word frequency
    words = words = tweet.split()
    freq = 0
    for word in words:
        freq += word_frequency(word, 'en')

    # avg
    freq = freq/len(words)
    frequency.append(freq*10000)

tweet_len = tweet_len * 27
readability = readability * 27
frequency = frequency * 27

df['length'] = tweet_len
df['readability'] = readability
df['frequency'] = frequency
# multiply frequency with 10000
df['frequency'] = df['frequency'] 

df = df.assign(src_length_class=lambda x: pd.cut(x.length, bins=[np.min(x.length), 15, 55, np.max(
    x.length)], labels=["short", "typical", "very long"], include_lowest=True))
df = df.assign(src_readability_class=lambda x: pd.cut(x.readability, bins=[np.min(
    x.readability), 50, 80, np.max(x.readability)], labels=["difficult", "fair", "easy"], include_lowest=True))
df = df.assign(src_frequency_class1=lambda x: pd.cut(x.frequency, bins=[np.min(x.frequency), 30, 50, np.max(
    x.frequency)], labels=["low-frequency", "regular-frequency", "high-frequency"], include_lowest=True))
df = df.assign(src_frequency_class2=lambda x: pd.cut(x.frequency, bins=[np.min(x.frequency), 50, 100, np.max(
    x.frequency)], labels=["low-frequency", "regular-frequency", "high-frequency"], include_lowest=True))
df = df.assign(src_frequency_class3=lambda x: pd.cut(x.frequency, bins=[np.min(x.frequency), 40, 80, np.max(
    x.frequency)], labels=["low-frequency", "regular-frequency", "high-frequency"], include_lowest=True))

opt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/csv4analysis/sota_reliability.csv'
df.to_csv(opt_path, encoding='utf-8', index=False)
