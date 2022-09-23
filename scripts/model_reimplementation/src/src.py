# reusable functions

def preprocess(text):
    # unified tweet preprocessing in TWEETEVAL: anonymize user mentions, remove urls, remove line breaks
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def load_tweets(file_path):
    # load tweets from text files
    tweets = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = preprocess(line)
            tweets.append(line)
    return tweets


def load_labels(file_path):
    # load labels from txt to pd of ints
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = int(line)
            labels.append(line)
    return labels
