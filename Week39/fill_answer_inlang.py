import os
import pandas as pd
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
)
import torch

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]
train_csv = os.path.join(filepath, "train_ar_ko_te_fil.csv")
val_csv   = os.path.join(filepath, "val_ar_ko_te_fil.csv")

df_train = pd.read_csv(train_csv)
df_val   = pd.read_csv(val_csv)

train_te = df_train[df_train["lang"] == "te"].copy()
val_te   = df_val[df_val["lang"] == "te"].copy()

train_te["answerable"] = train_te["answerable"].astype(bool)
val_te["answerable"]   = val_te["answerable"].astype(bool)

train_ans     = train_te[train_te["answerable"] == True].copy()
train_not_ans = train_te[train_te["answerable"] == False].copy()
val_ans       = val_te[val_te["answerable"] == True].copy()
val_not_ans   = val_te[val_te["answerable"] == False].copy()

print(f"Total TRAIN (te): {len(train_te)} | answerable={len(train_ans)} | unanswerable={len(train_not_ans)}")
print(f"Total VAL   (te): {len(val_te)} | answerable={len(val_ans)} | unanswerable={len(val_not_ans)}")
print("\nAnswerable rows with Telugu answers present/missing:")
print(f"TRAIN answerable: with_te={train_ans['answer_inlang'].notna().sum()} | missing={train_ans['answer_inlang'].isna().sum()}")
print(f"VAL   answerable: with_te={val_ans['answer_inlang'].notna().sum()} | missing={val_ans['answer_inlang'].isna().sum()}")

use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

for df in (train_te, val_te):
    df["had_good_te"] = df["answer_inlang"].notna() & (df["answer_inlang"].astype(str).str.strip() != "")

need_train_mt = train_ans[train_ans["answer_inlang"].isna()].copy()
need_val_mt   = val_ans[val_ans["answer_inlang"].isna()].copy()

train_te["will_be_pseudo"] = False
val_te["will_be_pseudo"]   = False
train_te.loc[need_train_mt.index, "will_be_pseudo"] = True
val_te.loc[need_val_mt.index,   "will_be_pseudo"]   = True

print(f"\nNLLB translation needs: TRAIN={len(need_train_mt)} | VAL={len(need_val_mt)}")

nllb_name = "facebook/nllb-200-distilled-600M"
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name, dtype=dtype, low_cpu_mem_usage=True)
nllb_tok = AutoTokenizer.from_pretrained(nllb_name)

translator = pipeline(
    "translation",
    model=nllb_model,
    tokenizer=nllb_tok,
    device=0 if use_cuda else -1,
    dtype="auto",
    max_length=256
)

def batch_translate_en_to_te(texts, batch_size=12):
    src_lang = "eng_Latn"
    tgt_lang = "tel_Telu"
    out = translator(texts, src_lang=src_lang, tgt_lang=tgt_lang, batch_size=batch_size)
    return [t["translation_text"] for t in out]

if len(need_train_mt) > 0:
    te_ans = batch_translate_en_to_te(need_train_mt["answer"].astype(str).tolist())
    train_te.loc[need_train_mt.index, "answer_inlang"] = te_ans

if len(need_val_mt) > 0:
    te_ans = batch_translate_en_to_te(need_val_mt["answer"].astype(str).tolist())
    val_te.loc[need_val_mt.index, "answer_inlang"] = te_ans

aug_train_csv = os.path.join(filepath, "train_te_augmented.csv")
aug_val_csv   = os.path.join(filepath, "val_te_augmented.csv")
train_te.to_csv(aug_train_csv, index=False, encoding="utf-8")
val_te.to_csv(aug_val_csv, index=False, encoding="utf-8")
print(f"\nSaved augmented Telugu-only CSVs:\n  {aug_train_csv}\n  {aug_val_csv}")