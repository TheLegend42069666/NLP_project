import os
import gc
import pandas as pd
import numpy as np
import torch
import sacrebleu
from sacrebleu.metrics import BLEU

from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer, MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]
train_csv = os.path.join(filepath, "train_te_augmented.csv")
val_csv   = os.path.join(filepath, "val_te_augmented.csv")
df_train = pd.read_csv(train_csv)
df_val   = pd.read_csv(val_csv)

def to_bool_series(s):
    return s.astype(str).str.strip().str.lower().isin(["true","1","t","yes","y"])

def has_col(df, name): return name in df.columns

train_te = df_train[df_train["lang"] == "te"].copy()
val_te   = df_val[df_val["lang"] == "te"].copy()

for df in (train_te, val_te):
    if has_col(df, "answerable"):
        df["answerable"] = to_bool_series(df["answerable"])
    else:
        df["answerable"] = True

def ensure_flags(df):
    if has_col(df, "had_good_te"):
        df["had_good_te"] = to_bool_series(df["had_good_te"])
    if has_col(df, "will_be_pseudo"):
        df["will_be_pseudo"] = to_bool_series(df["will_be_pseudo"])
    if (not has_col(df, "had_good_te")) or (not has_col(df, "will_be_pseudo")):
        ans_te_present = df["answer_inlang"].notna() & (df["answer_inlang"].astype(str).str.strip() != "")
        if not has_col(df, "will_be_pseudo"):
            df["will_be_pseudo"] = ~ans_te_present
        if not has_col(df, "had_good_te"):
            df["had_good_te"] = ans_te_present & (~df["will_be_pseudo"])
    return df

train_te = ensure_flags(train_te)
val_te   = ensure_flags(val_te)

train_ans     = train_te[train_te["answerable"]].copy()
train_not_ans = train_te[~train_te["answerable"]].copy()
val_ans       = val_te[val_te["answerable"]].copy()
val_not_ans   = val_te[~val_te["answerable"]].copy()

print("=== Data overview (Telugu) ===")
print(f"Total TRAIN (te): {len(train_te)} | answerable={len(train_ans)} | unanswerable={len(train_not_ans)}")
print(f"Total VAL   (te): {len(val_te)} | answerable={len(val_ans)} | unanswerable={len(val_not_ans)}")
print("\nAnswerable rows with Telugu answers present/missing:")
print(f"TRAIN answerable: with_te={(train_ans['answer_inlang'].notna() & (train_ans['answer_inlang'].astype(str).str.strip()!='')).sum()} | missing={(~train_ans['answer_inlang'].notna()).sum()}")
print(f"VAL   answerable: with_te={(val_ans['answer_inlang'].notna() & (val_ans['answer_inlang'].astype(str).str.strip()!='')).sum()} | missing={(~val_ans['answer_inlang'].notna()).sum()}")

def flag_counts(df, name):
    hg = int(df["had_good_te"].sum()) if "had_good_te" in df.columns else -1
    ps = int(df["will_be_pseudo"].sum()) if "will_be_pseudo" in df.columns else -1
    print(f"{name}: had_good_te={hg} | will_be_pseudo={ps}")

print("\nFlag counts (ANSWERABLE only):")
flag_counts(train_ans, "TRAIN answerable")
flag_counts(val_ans, "VAL   answerable")

train_good = train_ans[train_ans["had_good_te"]].copy()
train_pseudo = train_ans[train_ans["will_be_pseudo"] & ~train_ans["had_good_te"]].copy()

val_good = val_ans[val_ans["had_good_te"]].copy()
val_pseudo = val_ans[val_ans["will_be_pseudo"] & ~val_ans["had_good_te"]].copy()
val_unans = val_te[~val_te["answerable"]].copy()

good_up   = 8
pseudo_up = 1

train_all = pd.concat(
    [pd.concat([train_good]*good_up, ignore_index=True),
     pd.concat([train_pseudo]*pseudo_up, ignore_index=True)],
    ignore_index=True
).sample(frac=1.0, random_state=42).reset_index(drop=True)

print("\nTRAIN split sizes (after filter + upsample):")
print(f"  good (x{good_up}):     {len(train_good)*good_up}")
print(f"  pseudo (x{pseudo_up}): {len(train_pseudo)*pseudo_up}")
print(f"  TOTAL:                 {len(train_all)}")

print("\nVAL split sizes:")
print(f"  good:   {len(val_good)}")
print(f"  pseudo: {len(val_pseudo)}")
print(f"  unans:  {len(val_unans)}")

id2que_good   = val_good["question"].astype(str).to_dict()
id2que_pseudo = val_pseudo["question"].astype(str).to_dict()
id2que_unans  = val_unans["question"].astype(str).to_dict()

model_name = "google/mt5-base"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
model.to(device)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = tokenizer.eos_token_id
if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.pad_token_id
model.config.use_cache = False

for p in model.parameters():
    p.requires_grad = False

lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
    bias="none", task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_cfg)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

trainable_params = 0
all_params = 0
for p in model.parameters():
    n = p.numel(); all_params += n
    if p.requires_grad: trainable_params += n
print(f"\ntrainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100*trainable_params/all_params:.4f}")

def row_to_io(row):
    que = str(row["question"]).strip()
    con = str(row["context"]).strip()
    tgt_raw = row.get("answer_inlang", None)
    if tgt_raw is None or pd.isna(tgt_raw): return None, None
    tgt = str(tgt_raw).strip()
    if tgt == "" or tgt.lower() == "nan": return None, None
    src = f"task: qa\nlang: te\nAnswer in fluent Telugu using ONLY the context.\nquestion: {que}\ncontext: {con}\n"
    return src, tgt

def build_hf_dataset(df, tok, max_src_len=768, max_tgt_len=64, keep_id=False):
    srcs, tgts, ids = [], [], []
    for idx, r in df.iterrows():
        s, t = row_to_io(r)
        if s is None: continue
        srcs.append(s); tgts.append(t)
        if keep_id: ids.append(int(idx))
    enc = tok(srcs, truncation=True, max_length=max_src_len, padding=False)
    dec = tok(text_target=tgts, truncation=True, max_length=max_tgt_len, padding=False)
    pad = tok.pad_token_id
    labels = [[(x if x != pad else -100) for x in seq] for seq in dec["input_ids"]]
    enc["labels"] = labels
    if keep_id: enc["example_id"] = ids
    return Dataset.from_dict(enc)

ds_train      = build_hf_dataset(train_all, tokenizer, 1024, 128)
ds_val_good   = build_hf_dataset(val_good, tokenizer, 1024, 128, keep_id=True)
ds_val_pseudo = build_hf_dataset(val_pseudo, tokenizer, 1024, 128, keep_id=True)
ds_val_unans  = build_hf_dataset(val_unans, tokenizer, 1024, 128, keep_id=True)

print("\nHF Datasets built:")
print(ds_train); print(ds_val_good); print(ds_val_pseudo); print(ds_val_unans)

out_dir = os.path.join(filepath, "mt5base_te_qa_lora2")

gen_kwargs = dict(
    num_beams=6,
    length_penalty = 0.8,
    min_new_tokens=1,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15
)
model.generation_config.update(**gen_kwargs)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id

training_args = Seq2SeqTrainingArguments(
    output_dir=out_dir,
    do_train=True,
    overwrite_output_dir=True,
    optim="adamw_torch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=12,
    generation_num_beams=6,
    learning_rate=4e-4,
    lr_scheduler_type="constant_with_warmup",
    warmup_ratio=0.06,
    weight_decay=0.0,
    label_smoothing_factor=0.0,
    max_grad_norm=1.0,
    num_train_epochs=16,
    predict_with_generate=True,
    generation_max_length=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_chrf",
    greater_is_better=True,
    logging_steps=50,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    report_to="none",
    gradient_checkpointing=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

def _safe_batch_decode(ids_like, tok):
    if isinstance(ids_like, tuple):
        ids_like = ids_like[0]
    arr = np.asarray(ids_like)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    vocab_size = getattr(tok, "vocab_size", None) or len(tok)
    arr = np.where((arr < 0) | (arr >= vocab_size), pad_id, arr).astype(np.int64)
    return tok.batch_decode(arr.tolist(), skip_special_tokens=True)

def compute_metrics(eval_pred):
    pred_ids, labels = eval_pred
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)
    preds = _safe_batch_decode(pred_ids, tokenizer)
    refs  = _safe_batch_decode(labels, tokenizer)
    bleu1 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=1).corpus_score(preds, [refs]
    ).score
    bleu2 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=2).corpus_score(preds, [refs]
    ).score
    bleu3 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=3).corpus_score(preds, [refs]
        ).score
    bleu4 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=4).corpus_score(preds, [refs]
        ).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    denom = (sum(len(r.split()) for r in refs) + 1e-9)
    len_ratio = (sum(len(p.split()) for p in preds) + 1e-9)/denom
    return {"bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4, 
            "chrf": chrf, "len_ratio": len_ratio, "eval_chrf": chrf}

ds_val_eval = concatenate_datasets([ds_val_good.remove_columns(["example_id"]),
    ds_val_pseudo.remove_columns(["example_id"])])

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val_eval,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\n=== Training starts ===")
trainer.train()
print("=== Training ends ===\n")

merge_dir = os.path.join(filepath, "mt5base_te_qa_merged")

with torch.no_grad():
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merge_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merge_dir)
    merged.generation_config.save_pretrained(merge_dir)

def decode_preds(pred_ids, tok): return _safe_batch_decode(pred_ids, tok)
def decode_labels(label_ids, tok): return _safe_batch_decode(label_ids, tok)

def eval_bucket(name, preds_obj, tok):
    preds = decode_preds(preds_obj.predictions, tok)
    refs  = decode_labels(preds_obj.label_ids, tok)
    if len(preds) == 0:
        print(f"\n{name}: 0 samples, skipping."); return
    bleu1 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=1).corpus_score(preds, [refs]
    ).score
    bleu2 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=2).corpus_score(preds, [refs]
    ).score
    bleu3 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=3).corpus_score(preds, [refs]
        ).score
    bleu4 = BLEU(
        tokenize="none", 
        effective_order=True, 
        max_ngram_order=4).corpus_score(preds, [refs]
        ).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    pred_len = sum(len(p.split()) for p in preds) + 1e-9
    ref_len  = sum(len(r.split()) for r in refs) + 1e-9
    len_ratio = pred_len / ref_len
    print(
        f"\n{name} [{len(preds)}]: "
        f"BLEU1={bleu1:.2f} | BLEU2={bleu2:.2f} | BLEU3={bleu3:.2f} | BLEU4={bleu4:.2f} | "
        f"chrF={chrf:.2f} | len-ratio={len_ratio:.2f}"
    )

def show_samples(name, preds_obj, tok, ids, id2q, k=3):
    preds = decode_preds(preds_obj.predictions, tok)
    refs  = decode_labels(preds_obj.label_ids, tok)
    k = min(k, len(preds), len(ids))
    print(f"\n{name} samples:")
    for i in range(k):
        ex_id = ids[i]
        que = id2q.get(ex_id, "<missing>")
        print(f"--- {i+1} ---")
        print("Q:  ", que)
        print("PRED:", preds[i])
        print("REF: ", refs[i])

print("\n=== Final evaluation ===")
with torch.inference_mode():
    pred_good   = trainer.predict(ds_val_good)
    pred_pseudo = trainer.predict(ds_val_pseudo)
    pred_unans  = trainer.predict(ds_val_unans)

eval_bucket("VAL good",   pred_good,   tokenizer)
eval_bucket("VAL pseudo", pred_pseudo, tokenizer)
eval_bucket("VAL unans",  pred_unans,  tokenizer)

good_ids   = ds_val_good["example_id"]
pseudo_ids = ds_val_pseudo["example_id"]
unans_ids  = ds_val_unans["example_id"]

print("\nSample predictions from VAL good:")
show_samples("VAL good",   pred_good,   tokenizer, good_ids,   id2que_good,   k=7)
print("\nSample predictions from VAL pseudo:")
show_samples("VAL pseudo", pred_pseudo, tokenizer, pseudo_ids, id2que_pseudo, k=3)
print("\nSample predictions from VAL unanswerable (proxy refs):")
show_samples("VAL unans",  pred_unans,  tokenizer, unans_ids,  id2que_unans,  k=3)

if use_cuda:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
gc.collect()

# results:
# VAL good [7]: BLEU1=30.00 | BLEU2=22.36 | BLEU3=23.21 | BLEU4=23.21 | chrF=38.23 | len-ratio=1.25

# VAL pseudo [284]: BLEU1=9.57 | BLEU2=6.08 | BLEU3=3.87 | BLEU4=2.52 | chrF=15.46 | len-ratio=0.58

# VAL unans [93]: BLEU1=5.28 | BLEU2=2.31 | BLEU3=1.92 | BLEU4=1.88 | chrF=11.14 | len-ratio=0.86

# Sample predictions from VAL good:

# VAL good samples:
# --- 1 ---
# Q:   మున్నా చిత్రానికి సంగీత దర్శకుడు ఎవరు?
# PRED: ప్రకాష్ జయరాజ్
# REF:  హరీష్ జైరాజ్
# --- 2 ---
# Q:   విశ్వామిత్రుడు ఏ స్వర్గాన్ని నిర్మించాడు?
# PRED: త్రిశంఖు
# REF:  త్రిశంకు
# --- 3 ---
# Q:   సింగిరెడ్డి నారాయణరెడ్డి జ్ఞానపీఠ పురస్కారం ను ఎప్పుడు అందుకున్నాడు ?
# PRED: జులై 29, 1931
# REF:  1988
# --- 4 ---
# Q:   2011 జనగణన ప్రకారం గొట్టిప్రోలు గ్రామములో ఎన్ని ఇళ్లులు ఉన్నాయి?
# PRED: 511
# REF:  511
# --- 5 ---
# Q:   2011 జనగణన ప్రకారం పెదలోవ గ్రామములో ఎన్ని ఇళ్లులు ఉన్నాయి?
# PRED: 23
# REF:  23
# --- 6 ---
# Q:   2011 జనగణన ప్రకారం రెయ్యలగడ్ద గ్రామములో పురుషుల సంఖ్య ఎంత?
# PRED: 76
# REF:  37
# --- 7 ---
# Q:   2011 జనాభా లెక్కల ప్రకారం బూతుమిల్లిపాడు గ్రామ జనాభా ఎంత ?
# PRED: 433
# REF:  433

# Sample predictions from VAL pseudo:

# VAL pseudo samples:
# --- 1 ---
# Q:   ఒరెగాన్ రాష్ట్రంలోని అతిపెద్ద నగరం ఏది ?
# PRED: పాల్టన్
# REF:  పోర్ట్లాండ్
# --- 2 ---
# Q:   కలరా వ్యాధిని మొదటగా ఏ దేశంలో కనుగొన్నారు ?
# PRED: 1817
# REF:  భారతీయ ఉపఖండం
# --- 3 ---
# Q:   కలరా వ్యాధిని మొదటగా ఏ దేశంలో కనుగొన్నారు ?
# PRED: భారతదేశం
# REF:  ఇంగ్లాండ్

# Sample predictions from VAL unanswerable (proxy refs):

# VAL unans samples:
# --- 1 ---
# Q:   మలేరియా వ్యాధి కి మందు కనిపెట్టిన శాస్త్రవేత్త ఎవరు?
# PRED: రాబర్ట్ రోస్
# REF:  హన్స్ ఆండర్సాగ్
# --- 2 ---
# Q:   ఈస్ట్ ఇండియా కంపెనీ భారతదేశంలోకి ఎప్పుడు వచ్చింది?
# PRED: 1600
# REF:  1608
# --- 3 ---
# Q:   తెలుగు పంచాంగం ప్రకారం నూతన సంవత్సరం ఏ ఇంగ్లీష్ నెలలో ప్రారంభమవుతుంది?
# PRED: యుగాది
# REF:  మార్చి లేదా ఏప్రిల్
