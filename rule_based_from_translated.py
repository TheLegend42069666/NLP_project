import os
import pandas as pd
import regex as re
from nltk.corpus import stopwords

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil_tran.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil_tran.csv"))

langs = ["ar", "ko", "te"]

stopwords = set(stopwords.words("english"))

def normalize(text):
    return re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", str(text).lower())

def content_words(text):
    tokens = normalize(text).split()
    return [t for t in tokens if t not in stopwords]

num_re   = re.compile(r"\d+")
year_re  = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
where_re = re.compile(r"\b(in|at|on|from)\s+[A-Z][A-Za-z]+\b")
name_re  = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

def any_bigram_in_context(que_tokens, con_tokens):
    if len(que_tokens) < 2 or len(con_tokens) < 2:
        return False
    c_bigrams = set(zip(con_tokens, con_tokens[1:]))
    return any(bg in c_bigrams for bg in zip(que_tokens, que_tokens[1:]))

def has_number_overlap(que_raw, con_raw):
    que_nums = set(num_re.findall(str(que_raw)))
    if not que_nums:
        return False
    con_nums = set(num_re.findall(str(con_raw)))
    return bool(que_nums & con_nums)

def classify_answerable(question, context, thresh, min_ratio):
    que_words = content_words(question)
    con_words = content_words(context)
    overlap = sum(1 for w in que_words if w in con_words)
    ratio = overlap / max(1, len(que_words))
    
    bigram_match = any_bigram_in_context(que_words, con_words)
    number_ok    = has_number_overlap(question, context)

    que_norm = " ".join(que_words)  # normalized lowercase question for simple startswith checks
    con_raw = str(context)
    
    wh_word_check = False
    if que_norm.startswith("how many"):
        wh_word_check = number_ok
    elif que_norm.startswith("when"):
        wh_word_check = bool(year_re.search(con_raw)) or number_ok
    elif que_norm.startswith("who"):
        wh_word_check = bool(name_re.search(con_raw))
    elif que_norm.startswith("where"):
        wh_word_check = bool(where_re.search(con_raw))
    
    return (overlap >= thresh) or (ratio >= min_ratio) or bigram_match or number_ok or wh_word_check

print("\nRule-based classifier performance (train-tuned threshold, eval on val):")
for i in langs:
    train_subset = df_train[df_train["lang"] == i]
    train_q = train_subset["question_en"].astype(str).tolist()
    train_y = train_subset["answerable"].tolist()

    best_thresh = None
    best_ratio = None
    best_acc = -1
    train_scores = {}

    for thresh in range(0, 11):
        for ratio in [0.10, 0.20, 0.30, 0.4, 0.5]:
            preds = [classify_answerable(q, c, thresh, ratio)
                    for q, c in zip(train_q, train_subset["context"].astype(str))]
            tp = sum(p and y for p, y in zip(preds, train_y))
            tn = sum((not p) and (not y) for p, y in zip(preds, train_y))
            acc = (tp + tn) / len(train_y) if len(train_y) else 0.0
            train_scores[thresh] = acc
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
                best_ratio = ratio

    val_subset = df_val[df_val["lang"] == i]

    val_que = val_subset["question_en"].astype(str).tolist()
    val_lab = val_subset["answerable"].tolist()

    preds = [classify_answerable(que, con, best_thresh, 0.30)
             for que, con in zip(val_que, val_subset["context"].astype(str))]

    tp = sum(pred and lab for pred, lab in zip(preds, val_lab))
    tn = sum((not pred) and (not lab) for pred, lab in zip(preds, val_lab))
    fp = sum(pred and (not lab) for pred, lab in zip(preds, val_lab))
    fn = sum((not pred) and lab for pred, lab in zip(preds, val_lab))

    acc = (tp + tn) / len(val_que)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0

    print(f"\nLanguage: {i}")
    print("Train acc by threshold and ratio:", {k: round(v, 3) for k, v in train_scores.items()})
    print(f"Treshold = {best_thresh}, ratio = {best_ratio} (train acc={best_acc:.3f})")
    print(f"samples: {len(val_lab)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")
