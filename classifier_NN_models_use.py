import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

filepath = r"C:/Users/kkove/Desktop/NLP_project"
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil.csv"))
langs    = ["ar", "ko", "te"]

use_cuda = torch.cuda.is_available()
device   = "cuda" if use_cuda else "cpu"

def load_model_and_tokenizer(lang):
    model_dir = os.path.join(filepath, f"distilmbert_cls_{lang}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model

def predict_one(lang, que, con, max_length=256):
    tokenizer, model = load_model_and_tokenizer(lang)
    enc = tokenizer(que, con, truncation=True, padding="max_length",
                    max_length=max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = int(np.argmax(probs))         # 1 = answerable
    return pred, float(probs[1])         # (label, P(answerable))

# Example single prediction:
# p, prob = predict_one("ar", "من هو مؤسس جوجل؟", "English context goes here ...")
# print(p, prob)

# ---- Batch predict on your validation set per language ----
df_val["pred_answerable"] = 0
df_val["prob_answerable"] = 0.0

for lang in langs:
    print(f"\nPredicting validation for lang = {lang}")
    val = df_val[df_val["lang"] == lang]

    if val.empty:
        continue

    que_val = val["question"].astype(str).tolist()
    con_val = val["context"].astype(str).tolist()
    lab_val = val["answerable"].astype(int).tolist()

    tokenizer, model = load_model_and_tokenizer(lang)

    preds_all, probs_all = [], []
    bs = 16 if use_cuda else 8
    for i in range(0, len(que_val), bs):
        batch_que = que_val[i:i+bs]
        batch_con = con_val[i:i+bs]
        enc = tokenizer(batch_que, batch_con, truncation=True, padding=True,
                        max_length=256, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1]
            preds  = torch.argmax(logits, dim=-1)
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    # write back to the same rows
    df_val.loc[val.index, "pred_answerable"] = preds_all
    df_val.loc[val.index, "prob_answerable"] = probs_all

    # quick metrics
    acc  = accuracy_score(lab_val, preds_all)
    prec, rec, f1, _ = precision_recall_fscore_support(lab_val, preds_all, average="binary", zero_division=0)
    print(f"\nLanguage: {lang}")
    print(f"VAL samples: {len(lab_val)}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
