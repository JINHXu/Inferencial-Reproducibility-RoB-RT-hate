# create data for significance testing
# Jinghua Xu

import pandas as pd
from collections import Counter
import string
import textstat
import math
# from readability import Readability
from wordfreq import word_frequency
from src.src import load_tweets, load_labels, preprocess

# file paths
baseline_preds_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds.txt'
sota_preds_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/sota_preds.txt'

test_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_text.txt'
true_labels_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_labels.txt'

# load data
baseline_preds = load_labels(baseline_preds_path)
sota_preds = load_labels(sota_preds_path)
tweets = load_tweets(test_txt_path)
true_labels = load_labels(true_labels_path)

# get word frequency in training data
train_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/train_text.txt'
train_txt = load_tweets(train_txt_path)

vocab = []
for txt in train_txt:
    vocab += txt.split()

freq_dict = Counter(vocab)
len_vocab = len(vocab)

# tweet id
tweet_id = list(range(len(tweets)))

# tweet2id
tweet2id = dict(zip(tweets, tweet_id))

# tweet length
tweet_len = []

# Flesch-Kincaid readability
readability = []

# # word rarity
# rarity = []

# word frequency
frequency = []

for tweet in tweets:
    tweet_len.append(len(tweet.split()))

    # Flesch-Kincaid readability
    readability.append(textstat.flesch_reading_ease(tweet))

    # # Word rarity: summation of negative log frequency of each word in the sentence
    # words = tweet.split()
    # sentecen_rarity = 0
    # for word in words:
    #     if word[0] == '#':
    #         word = word[1:]
    #     word_freq = word_frequency(word, 'en')
    #     if word_freq == 0:
    #         continue
    #     sentence_rarity = sentecen_rarity - math.log(word_freq)

    # rarity.append(sentecen_rarity)

    # word frequency
    words = words = tweet.split()
    freq = 0
    for word in words:
        if word not in freq_dict.keys():
            continue
        freq += freq_dict[word]/len_vocab
    # avg
    freq = freq/len(words)
    frequency.append(freq)

# sota & baseline perf columns
sota_perf = []
baseline_perf = []
for i in range(len(true_labels)):

    if sota_preds[i] == true_labels[i]:
        sota_perf.append(1)
    else:
        sota_perf.append(0)

    if baseline_preds[i] == true_labels[i]:
        baseline_perf.append(1)
    else:
        baseline_perf.append(0)


# tweet id column
tweet_id = tweet_id+tweet_id

# tweet column
tweets = tweets + tweets

# system column
system = ['Baseline']*2970 + ['SoTA']*2970

# performance
perf = baseline_perf+sota_perf

# tweet length
tweet_len = tweet_len + tweet_len

# readability
readability = readability + readability

# # word rarity
# rarity = rarity + rarity

# frequency
frequency = frequency + frequency

df = pd.DataFrame({'tweet_id': tweet_id, 'system': system, 'performance': perf, # 'cleaned_tweet': tweets,
                   'tweet_length': tweet_len, 'frequency': frequency, 'readability': readability})  # 'word_rarity': rarity}

opt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/csv4analysis/baseline_sota.csv'
df.to_csv(opt_path, sep='\t', encoding='utf-8')