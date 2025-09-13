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
            if token not in seen:
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
    train_ngrams, vocab_prep = padded_everygram_pipeline(n, tokens_oov)
    lm = KneserNeyInterpolated(n)
    lm.fit(train_ngrams, vocab_prep)
    return lm, vocab

def logprob(lm, tokens, n):
    logp = 0.0; cnt = 0
    for ng in everygrams(tokens, max_len=n):
        last, hist = ng[-1], ng[:-1]
        logp += math.log(lm.score(last, hist) + 1e-12)  # avoid -inf
        cnt += 1
    return logp, cnt

def perplexity(lm, texts, vocab, n):
    tokens = [tokenize(t) for t in texts]
    tokens = [[w if w in vocab else "[OOV]" for w in row] for row in tokens]  # slides: OOV replace
    total_lp = 0.0; total_cnt = 0
    for row in tokens:
        lp, c = logprob(lm, row, n)
        total_lp += lp; total_cnt += max(1, c)
    return math.exp(-total_lp / total_cnt)

for lang in langs:
    train_subset = df_train[df_train["lang"] == f"{lang}"]
    train_ques = train_subset["question"]
    val_subset = df_val[df_val["lang"] == f"{lang}"]
    val_ques = val_subset["question"]

    tr_que_list = train_ques.astype(str).tolist()
    val_que_list = val_ques.astype(str).tolist()
    lm, vocab = train_ngram_model(tr_que_list, 3)
    pp = perplexity(lm, val_que_list, vocab, 3)

    print(f"pp for {lang}: {pp}")
    