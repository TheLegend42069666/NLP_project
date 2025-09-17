import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import gc

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil.csv"))

langs = ["ar", "ko", "te"]

use_cuda = torch.cuda.is_available()
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_pairs(df):
    que = df["question"].astype(str).tolist()
    con = df["context"].astype(str).tolist()
    lab = df["answerable"].astype(int).tolist()
    return que, con, lab

def tokenize_pairs(que_list, con_list, max_length=256):
    return tokenizer(
        que_list,
        con_list,
        truncation=True,
        padding="max_length",
        max_length=max_length,     # keep modest to avoid OOM
        return_tensors=None        # return Python lists (Dataset will handle tensors)
    )

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

for lang in langs:
    print(f"\n===== Fine-tuning DistilBERT for language: {lang} =====")

    # Split
    train = df_train[df_train["lang"] == lang]
    val   = df_val[df_val["lang"] == lang]

    que_train, con_train, lab_train = make_pairs(train)
    que_val,   con_val,   lab_val   = make_pairs(val)

    # Tokenize -> dicts of lists
    enc_train = tokenize_pairs(que_train, con_train, max_length=256)
    enc_val   = tokenize_pairs(que_val,   con_val,   max_length=256)

    # Build HF Datasets (no custom class)
    enc_train["labels"] = lab_train
    enc_val["labels"]   = lab_val
    ds_train = Dataset.from_dict(enc_train)
    ds_val   = Dataset.from_dict(enc_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        num_labels=2
    )

    # Training config
    out_dir = os.path.join(filepath, f"distilmbert_cls_{lang}")
    args = TrainingArguments(
        output_dir=out_dir,
        # evaluation_strategy="epoch",
        # save_strategy="no",
        # load_best_model_at_end=True,
        # metric_for_best_model="accuracy",
        # greater_is_better=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8 if use_cuda else 4,
        per_device_eval_batch_size=16 if use_cuda else 8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=42,
        fp16=use_cuda,   # enable AMP on GPU
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    print(f"\nLanguage: {lang}")
    print(f"VAL samples: {len(lab_val)}")
    print(f"Accuracy:  {eval_metrics["eval_accuracy"]:.3f}")
    print(f"Precision: {eval_metrics["eval_precision"]:.3f}")
    print(f"Recall:    {eval_metrics["eval_recall"]:.3f}")
    print(f"F1:        {eval_metrics["eval_f1"]:.3f}")

    del trainer, model, ds_train, ds_val, enc_train, enc_val
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# result:

# Language: ar
# VAL samples: 415
# Accuracy:  0.981
# Precision: 0.994
# Recall:    0.983
# F1:        0.989

# Language: ko
# VAL samples: 356
# Accuracy:  0.955
# Precision: 0.955
# Recall:    1.000
# F1:        0.977

# Language: te
# VAL samples: 384
# Accuracy:  0.930
# Precision: 0.923
# Recall:    0.990
# F1:        0.955