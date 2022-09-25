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
baseline_preds_path1 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds1.txt'
sota_preds_path1 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/sota_preds1.txt'

baseline_preds_path2 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds2.txt'
sota_preds_path2 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/sota_preds2.txt'

baseline_preds_path3 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds3.txt'
sota_preds_path3 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/sota_preds3.txt'

test_txt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_text.txt'
true_labels_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/hate-eval/test_labels.txt'

# load data
baseline_preds1 = load_labels(baseline_preds_path1)
sota_preds1 = load_labels(sota_preds_path1)

baseline_preds2 = load_labels(baseline_preds_path2)
sota_preds2 = load_labels(sota_preds_path2)

baseline_preds3 = load_labels(baseline_preds_path3)
sota_preds3 = load_labels(sota_preds_path3)

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
sota_perf1 = []
baseline_perf1 = []
sota_perf2 = []
baseline_perf2 = []
sota_perf3 = []
baseline_perf3 = []

for i in range(len(true_labels)):

    if sota_preds1[i] == true_labels[i]:
        sota_perf1.append(1)
    else:
        sota_perf1.append(0)

    if baseline_preds1[i] == true_labels[i]:
        baseline_perf1.append(1)
    else:
        baseline_perf1.append(0)

    if sota_preds2[i] == true_labels[i]:
        sota_perf2.append(1)
    else:
        sota_perf2.append(0)

    if baseline_preds2[i] == true_labels[i]:
        baseline_perf2.append(1)
    else:
        baseline_perf2.append(0)

    if sota_preds3[i] == true_labels[i]:
        sota_perf3.append(1)
    else:
        sota_perf3.append(0)

    if baseline_preds3[i] == true_labels[i]:
        baseline_perf3.append(1)
    else:
        baseline_perf3.append(0)

# tweet id column
tweet_id = tweet_id * 6

# tweet column
tweets = tweets * 6

# system column
system = ['Baseline']*2970*3 + ['SoTA']*2970*3

# performance
perf = baseline_perf1 + baseline_perf2 + \
    baseline_perf3 + sota_perf1 + sota_perf2 + sota_perf3

# tweet length
tweet_len = tweet_len * 6

# readability
readability = readability * 6

# # word rarity
# rarity = rarity + rarity

# frequency
frequency = frequency * 6

df = pd.DataFrame({'tweet_id': tweet_id, 'system': system, 'performance': perf,  # 'cleaned_tweet': tweets,
                   'length': tweet_len, 'frequency': frequency, 'readability': readability})  # 'word_rarity': rarity}

opt_path = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/csv4analysis/baseline_sota_repeated.csv'
df.to_csv(opt_path, encoding='utf-8', index=False)