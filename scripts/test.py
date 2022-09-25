from src.src import load_tweets, load_labels, preprocess

p1 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds1.txt'
p2 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds2.txt'
p3 = '/Users/xujinghua/Inferencial-Reproducibility-RoB-RT-hate/data/predictions/svm_preds3.txt'

l1 = load_labels(p1)
l2 = load_labels(p2)
l3 = load_labels(p3)

for i in range(len(l1)):
    if l3[i] != l1[i]:
        print(l3[i])
        print(l1[i])