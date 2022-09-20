# Inferencial Reproducibility Examination on [RoB-RT-hate]() ([Barbieri et al., 2020]())

_Term project for HS Empirical Methods for NLP and Data Science (SS22 Uni HD)_

**Supervisor**: Prof. Dr. Stafan Riezler

**Author**: Jinghua Xu          


This project intends to examine the inferencial reproducibility of the SOTA model proposed in [Barbieri et al., 2020]() for hate speech detection. 

In order to measure the inferential reproducibility of the SOTA model, this analysis conducts [Statistical Significance Test using Linear Mixed Effect Models]() and [Reliability Test of Model Prediction Performance]().

The baseline is an SVM-based approach using both word and character n-gram features, a model and feature set that has seen great success in Twitter-based shared tasks such as emoji prediction ([Çöltekin & Rama, 2018]()) and stance prediction ([Mohammad et al., 2018]()). 

The 

**The following lists the data properties, meta-parameters considered in this analysis:**

### Measuring Data: data properties to consider

- Word rarity ([Platanios et al., 2019]()): Negative log of empirical probabilities of words in segment, higher value means higher rarity.

- Flesch-Kincaid readability ([Kincaid et al., 1975]()): Pro-rates words/sentences and syllables/word; in principle unbounded, but interpretation scheme exists for ranges from 0 (difficult) to 100 (easy).

- Sentence Length: Number of tokens in each preprocessed/cleaned tweet.[^footnote]

### Meta-parameters?

* learning rate?
* 

### Random Seeds?

### 

## References

Riezler, Stefan, and Michael Hagmann. "Validity, Reliability, and Significance: Empirical Methods for NLP and Data Science." Synthesis Lectures on Human Language Technologies 14.6 (2021): 1-165.

[TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148) (Barbieri et al., Findings 2020)

[SemEval-2019 Task 5: Multilingual Detection of Hate Speech Against Immigrants and Women in Twitter](https://aclanthology.org/S19-2007) (Basile et al., SemEval 2019)

Platanios, Emmanouil Antonios, et al. "Competence-based curriculum learning for neural machine translation." arXiv preprint arXiv:1903.09848 (2019).

Peter Kincaid, Robert P. Fishburne Jr., Richard L. Rogers, and Brad S. Chissom. 1975. Derivation of New Readability Formulas (Automated Readability Index, Fog Count and Flesch Reading Ease Formula) for Navy Enlisted Personnel. Research Branch Report, Millington, TN: Chief of Naval Training.

[Tübingen-Oslo at SemEval-2018 Task 2: SVMs perform better than RNNs in Emoji Prediction](https://aclanthology.org/S18-1004) (Çöltekin & Rama, SemEval 2018)

[SemEval-2018 Task 1: Affect in Tweets](https://aclanthology.org/S18-1001) (Mohammad et al., SemEval 2018)

[^footnote]: Preprocessing...

