import os
import pandas as pd
import regex as re
from nltk.corpus import stopwords

filepath = r"C:/Users/kkove/Desktop/NLP_project"

# use the translated validation file
df_val = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil_tran.csv"))

langs = ["ar", "ko", "te"]

stopwords = set(stopwords.words("english"))

def normalize(text):
    return re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", str(text).lower())

def content_words(text):
    tokens = normalize(text).split()
    return [t for t in tokens if t not in stopwords]

def classify_answerable(question, context, thresh):
    que_words = content_words(question)
    con_words = set(content_words(context))
    overlap = sum(1 for w in que_words if w in con_words)
    # ratio = overlap / max(1, len(que_words))
    if overlap >= thresh:
        return True
    else:
        return False

# --- Evaluate on translated validation set ---
print("\nRule-based classifier performance:")
for i in langs:
    subset = df_val[df_val["lang"] == i]
    if subset.empty:
        print(f"\nLanguage: {i} has no samples, skipping.")
        continue

    que_eng = subset["question_en"].astype(str).tolist()

    corr_labels = subset["answerable"].tolist()

    scores = {}
    best_preds = None
    best_thresh = None

    for thresh in range(1, 11):
        preds = []
        for que, con in zip(que_eng, subset["context"].astype(str)):
            preds.append(classify_answerable(que, con, thresh))
        tp = sum(pred and lab for pred, lab in zip(preds, corr_labels))
        tn = sum((not pred) and (not lab) for pred, lab in zip(preds, corr_labels))
        acc = (tp+tn) / len(corr_labels) if corr_labels else 0
        scores[thresh] = acc
        if best_thresh is None or acc >= scores[best_thresh]:
            best_preds = preds.copy()
            best_thresh = thresh

    tp = sum(pred and lab for pred, lab in zip(best_preds, corr_labels))
    tn = sum((not pred) and (not lab) for pred, lab in zip(best_preds, corr_labels))
    fp = sum(pred and (not lab) for pred, lab in zip(best_preds, corr_labels))
    fn = sum((not pred) and lab for pred, lab in zip(best_preds, corr_labels))

    acc = scores[best_thresh]
    prec = tp / (tp+fp) if (tp+fp) else 0
    rec  = tp / (tp+fn) if (tp+fn) else 0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0

    print(f"Language: {i}")
    print("Scores by threshold:", {k: round(v, 3) for k, v in scores.items()})
    print(f"Best threshold for {i}: {best_thresh}")
    print(f"Total samples: {len(corr_labels)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}\n")
