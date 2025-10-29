import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import torch
import gc

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]

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
        max_length=max_length,
        return_tensors=None
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
    print(f"\n=== Fine-tuning DistilBERT for language: {lang} ===")

    train = df_train[df_train["lang"] == lang]
    val   = df_val[df_val["lang"] == lang]

    que_train, con_train, lab_train = make_pairs(train)
    que_val,   con_val,   lab_val   = make_pairs(val)

    enc_train = tokenize_pairs(que_train, con_train, max_length=256)
    enc_val   = tokenize_pairs(que_val,   con_val,   max_length=256)

    enc_train["labels"] = lab_train
    enc_val["labels"]   = lab_val
    ds_train = Dataset.from_dict(enc_train)
    ds_val   = Dataset.from_dict(enc_val)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        num_labels=2
    )

    out_dir = os.path.join(filepath, f"distilmbert_cls_{lang}")
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=6,
        weight_decay=0.0,
        logging_steps=50,
        report_to="none",
        seed=42,
        bf16=use_cuda,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2
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
    print(f"Accuracy:  {eval_metrics['eval_accuracy']:.3f}")
    print(f"Precision: {eval_metrics['eval_precision']:.3f}")
    print(f"Recall:    {eval_metrics['eval_recall']:.3f}")
    print(f"F1:        {eval_metrics['eval_f1']:.3f}")

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    if lang == "te":
        te_model_dir = os.path.join(filepath, "distilmbert_cls_te")

        inf_tokenizer = AutoTokenizer.from_pretrained(te_model_dir)
        inf_model = AutoModelForSequenceClassification.from_pretrained(te_model_dir, num_labels=2)
        if use_cuda:
            inf_model.to("cuda")
        inf_model.eval()

        def classify_answerable(question_text: str, context_text: str) -> bool:
            with torch.no_grad():
                enc = inf_tokenizer(
                    question_text,
                    context_text,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors="pt"
                )
                enc = {k: v.to(inf_model.device) for k, v in enc.items()}
                logits = inf_model(**enc).logits
                return int(logits.argmax(dim=-1).item()) == 1

        df_test = pd.read_json(os.path.join(filepath, "test.json"))
        test_subset = df_test[df_test["lang"] == "te"].copy()

        test_preds = [
            classify_answerable(que, con)
            for que, con in zip(test_subset["question"].astype(str), test_subset["context"].astype(str))
        ]

        print("\nResults on test.json (te only):")
        for _id, pred in zip(test_subset["id"].tolist(), test_preds):
            print(f"id={_id} - predicted: {'answerable' if pred else 'unanswerable'}")

    del trainer, model, ds_train, ds_val, enc_train, enc_val
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

# results:

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