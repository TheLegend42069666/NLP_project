import os
import pandas as pd
import regex as re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.corpus import stopwords
from tqdm import trange
import torch

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil.csv"))

langs = ["ar", "ko", "te"]

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    dtype=dtype,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

target_lang = "eng_Latn"
langs_nllb = {"ar": "arb_Arab", "ko": "kor_Hang", "te": "tel_Telu"}

translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    device=0 if use_cuda else -1,
    dtype="auto",
    max_length=256
)

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

def batch_translate(texts, src_lang, tgt_lang, batch_size=8):
    results = []
    for i in trange(0, len(texts), batch_size, desc=f"Translating {src_lang} to {tgt_lang}"):
        batch = texts[i:i+batch_size]
        translations = translator(
            batch,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        results.extend([t["translation_text"] for t in translations])
    return results

# --- Evaluate on validation set ---
print("\nRule-based classifier performance:")
for i in langs:
    subset = df_val[df_val["lang"] == i]
    questions = subset["question"].astype(str).tolist()
    que_eng = batch_translate(
        questions, src_lang=langs_nllb[i], tgt_lang=target_lang, batch_size=16
    )

    preds = []
    max_pred = []
    scores = {}
    corr_labels = subset["answerable"].tolist()
    for thresh in range(1, 11):
        for que, con in zip(que_eng, subset["context"].astype(str)):
            preds.append(classify_answerable(que, con, thresh))
        tp = sum(pred and lab for pred, lab in zip(preds, corr_labels))
        scores[thresh] = tp
        if best_thresh is None or tp >= scores[best_thresh]:
            best_preds = preds.copy()
            best_thresh = thresh
        
        

    tp = scores[best_thresh]
    tn = sum((not pred) and (not lab) for pred, lab in zip(best_preds, corr_labels))
    fp = sum(pred and (not lab) for pred, lab in zip(best_preds, corr_labels))
    fn = sum((not pred) and lab for pred, lab in zip(best_preds, corr_labels))

    acc = (tp+tn) / len(corr_labels)
    prec = tp / (tp+fp) if (tp+fp) else 0
    rec = tp / (tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) else 0

    print(f"\nBest threshold for {i}: {best_thresh}")
    print(f"\nLanguage: {i}")
    print(f"Total samples: {len(corr_labels)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1: {f1:.3f}")
