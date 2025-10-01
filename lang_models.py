import os
import pandas as pd
import regex as re
import math
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline, pad_both_ends
from nltk.util import ngrams

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil_tran.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil_tran.csv"))

langs = ["ar", "ko", "te"]

def tokenize(text):
    return re.findall(r"(?:\p{L}\p{M}*)(?:['’](?:\p{L}\p{M}*))+|\p{L}\p{M}*|\p{N}+", str(text))

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
    seq = list(pad_both_ends(tokens, n))
    logp = 0.0; count = 0
    for ng in ngrams(seq, n):
        last, hist = ng[-1], ng[:-1]
        logp += math.log(lm.score(last, hist) + 1e-12)
        count += 1
    return logp, count

def perplexity(lm, texts, vocab, n):
    tokens = [tokenize(t) for t in texts]
    tokens = [[w if w in vocab else "[OOV]" for w in row] for row in tokens]
    total_logp = 0.0; total_count = 0
    for row in tokens:
        logp, count = logprob(lm, row, n)
        total_logp += logp; total_count += max(1, count)
    return math.exp(-total_logp / total_count)

def oov_rate(texts, vocab):
    tokens = sum((tokenize(t) for t in texts), [])
    oov = sum(w not in vocab for w in tokens)
    return oov / max(1, len(tokens))

def token_count(texts): 
    return sum(len(tokenize(t)) for t in texts)

for lang in ["ar","ko","te"]:
    print(lang, "train Q tokens:", token_count(df_train[df_train["lang"]==lang]["question"]))
    print(lang, "val   Q tokens:", token_count(df_val[df_val["lang"]==lang]["question"]))
print("EN context train tokens:", token_count(df_train["context"]))
print("EN context val   tokens:", token_count(df_val["context"]))

n_grams = [4, 5]

for lang in langs:
    best_pp = None
    best_n = None
    for n in n_grams:
        print(f"Training {n}-gram model for {lang}...")
        train_subset = df_train[df_train["lang"] == f"{lang}"]
        train_ques = train_subset["question"]
        val_subset = df_val[df_val["lang"] == f"{lang}"]
        val_ques = val_subset["question"]

        train_que_list = train_ques.astype(str).tolist()
        val_que_list = val_ques.astype(str).tolist()
        lm, vocab = train_ngram_model(train_que_list, n)
        pp = perplexity(lm, val_que_list, vocab, n)
        if best_pp is None or pp < best_pp:
            best_pp = pp
            best_n = n

    print(f"{lang} - val OOV rate:", round(oov_rate(val_que_list, vocab), 4))
    print(f"{lang} - pp: {best_pp} (using {best_n}-gram)\n")

n_grams_eng = [4, 5]

for n in n_grams_eng:
    best_pp_eng = None
    best_n_eng = None
    print(f"Training {n}-gram model for English context...")
    con_train_list = df_train["context"].astype(str).tolist()[:63]
    con_val_list   = df_val["context"].astype(str).tolist()[:11]

    # train_subset = df_train[df_train["lang"] == f"ar"]
    # train_ques = train_subset["question"]
    # train_ques_list = train_ques.astype(str).tolist()
    # print(len(train_ques_list))

    # print(len(con_train_list), len(con_val_list))
    # print(con_train_list[0])

    lm_con, vocab_con = train_ngram_model(con_train_list, n)
    pp_con = perplexity(lm_con, con_val_list, vocab_con, n)
    if best_pp_eng is None or pp < best_pp_eng:
            best_pp_eng = pp
            best_n_eng = n

print("eng OOV rate:", round(oov_rate(con_val_list, vocab_con), 4))
print(f"pp for eng: {best_pp_eng} (using {best_n_eng}-gram)\n")
    