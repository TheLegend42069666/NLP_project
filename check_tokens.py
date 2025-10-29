import os, numpy as np, pandas as pd
from transformers import AutoTokenizer

from pathlib import Path
filepath = Path(__file__).resolve().parent
train_csv = os.path.join(filepath, "train_ar_ko_te_fil.csv")
val_csv   = os.path.join(filepath, "val_ar_ko_te_fil.csv")
model     = "xlm-roberta-base"

df = pd.concat([pd.read_csv(train_csv), pd.read_csv(val_csv)], ignore_index=True)
df["question"] = df["question"].astype(str)
df["context"]  = df["context"].astype(str)
df["answer"]   = df["answer"].astype(str)
df["answerable"] = df["answerable"].astype(bool)

tok = AutoTokenizer.from_pretrained(model, use_fast=True)

def token_lengths(texts, batch_size=512):
    lens = []
    for i in range(0, len(texts), batch_size):
        batch = list(map(str, texts[i:i+batch_size]))
        enc = tok(batch, add_special_tokens=False)
        lens.extend([len(ids) for ids in enc["input_ids"]])
    return np.array(lens, dtype=np.int32)

def summarize(name, arr):
    if len(arr)==0:
        print(f"{name}: (no data)"); return
    que = np.quantile(arr, [0.5, 0.95, 0.99]).astype(int)
    print(f"{name}: max={arr.max()}  mean={arr.mean():.1f}  p95={que[1]}  p99={que[2]}  (median={que[0]})")

def run_slice(df_slice, label):
    print(f"\n=== Stats: {label} ===")
    que_len = token_lengths(df_slice["question"])
    con_len = token_lengths(df_slice["context"])
    ans_len = token_lengths(df_slice.loc[df_slice["answerable"], "answer"])
    summarize("Question tokens", que_len)
    summarize("Context  tokens", con_len)
    summarize("Answer   tokens (answerable only)", ans_len)

    if len(ans_len) > 0:
        idx = df_slice.loc[df_slice["answerable"]].index
        topk_idx = idx[np.argsort(ans_len)[-5:]]
        print("\nLongest answers (token count, language, answer preview):")
        for i in topk_idx:
            alen = len(tok(df_slice.at[i, "answer"], add_special_tokens=False)["input_ids"])
            ans_preview = df_slice.at[i, "answer"][:120].replace("\n"," ")
            print(f"  {alen:>3} | {df_slice.at[i,'lang']} | {ans_preview}")

run_slice(df, "ALL LANGS")

for lang in ["ar", "ko", "te"]:
    run_slice(df[df["lang"]==lang], f"lang={lang}")

def estimate_pair_special_tokens(sample_n=200):
    samp = df.sample(min(sample_n, len(df)), random_state=42)
    total = 0
    for _, r in samp.iterrows():
        que_ids = tok(r["question"], add_special_tokens=False)["input_ids"]
        con_ids = tok(r["context"],  add_special_tokens=False)["input_ids"]
        pair  = tok(r["question"], r["context"], add_special_tokens=True)["input_ids"]
        total += (len(pair) - len(que_ids) - len(con_ids))
    return round(total / max(1,len(samp)))
print("\nApprox special tokens in pair:", estimate_pair_special_tokens())
