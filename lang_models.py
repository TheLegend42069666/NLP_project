import os
import pandas as pd
import regex as re
import math
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.util import everygrams

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil_tran.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil_tran.csv"))

langs = ["ar", "ko", "te"]

def tokenize(text):
    text = re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", str(text))
    tokens = text.split()
    return tokens

def oov_inject_train(tokens_list):
    seen = set()
    output = []
    for tokens in tokens_list:
        row = []
        for token in tokens:
            if token in seen:
                row.append("[OOV]")
                seen.add(token)
            else:
                row.append(token)
        output.append(row)
    return output, seen

def oov_replace(tokens_list, vocab):
    return [[word if word in vocab else "[OOV]"] for word in tokens_list]

def train_ngram_model(texts, n):
    tokens = [tokenize(text) for text in texts]
    tokens_oov, vocab = oov_inject_train(tokens)
    train_ngrams, padded_vocab = padded_everygram_pipeline(n, tokens_oov)
    lm = KneserNeyInterpolated(n)
    lm.fit(train_ngrams, padded_vocab)
    return lm, vocab

def