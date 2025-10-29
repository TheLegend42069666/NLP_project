import os
import json
import unicodedata
import numpy as np
import pandas as pd
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    DataCollatorForTokenClassification, TrainingArguments, Trainer
)
from seqeval.metrics import f1_score, precision_score, recall_score

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]
train_csv = os.path.join(filepath, "train_ar_ko_te_fil.csv")
val_csv   = os.path.join(filepath, "val_ar_ko_te_fil.csv")

def safe_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return default

def _normalize_for_search(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()

def add_char_spans(df: pd.DataFrame):
    starts, ends = [], []
    for _, r in df.iterrows():
        if not bool(r["answerable"]):
            starts.append(-1); ends.append(-1); continue
        con = str(r["context"])
        ans = str(r["answer"])
        s = safe_int(r.get("answer_start", -1), -1)
        e = safe_int(r.get("answer_end", -1), -1)
        if s >= 0 and e < 0:
            e = s + len(ans)
        ok = (s >= 0 and e >= 0 and e <= len(con) and con[s:e] == ans)
        if not ok:
            s2 = con.find(ans)
            if s2 == -1:
                s2 = con.lower().find(ans.lower())
            if s2 == -1:
                con_norm = _normalize_for_search(con)
                ans_norm = _normalize_for_search(ans)
                s_norm = con_norm.find(ans_norm)
                if s_norm != -1:
                    head = ans[: max(3, min(8, len(ans)))]
                    s3 = con.lower().find(head.lower())
                    if s3 != -1:
                        s2 = s3
            if s2 == -1:
                s = -1; e = -1
            else:
                s = s2; e = s2 + len(ans)
        starts.append(s); ends.append(e)
    df["answer_start"] = starts
    df["answer_end"]   = ends

def span_coverage_report(df: pd.DataFrame, name: str):
    tot_ans = int(df["answerable"].sum())
    miss = ((df["answerable"]) & ((df["answer_start"] < 0) | (df["answer_end"] < 0))).sum()
    print(f"[{name}] Answerable with missing spans: {miss}/{tot_ans} ({0 if tot_ans==0 else 100*miss/tot_ans:.1f}%)")


def build_windows_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer):
    all_input_ids, all_attn, all_labels, all_na = [], [], [], []

    for _, ex in df.iterrows():
        que = str(ex["question"]); con = str(ex["context"])
        ans = bool(ex["answerable"])
        s = int(ex["answer_start"]); e = int(ex["answer_end"])

        enc = tokenizer(
            que, con,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False
        )

        kept_pos = False
        n_feats = len(enc["input_ids"])

        for w in range(n_feats):
            input_ids = enc["input_ids"][w]
            attn      = enc["attention_mask"][w]
            offsets   = enc["offset_mapping"][w]
            seq_ids   = enc.sequence_ids(w)

            con_tok_idx = [i for i, sid in enumerate(seq_ids) if sid == 1]
            if not con_tok_idx:
                continue

            if (not ans) or s < 0 or e < 0:
                if w == 0:
                    labels = [(-100 if sid != 1 else label2id["O"]) for sid in seq_ids]
                    all_input_ids.append(input_ids); all_attn.append(attn); all_labels.append(labels); all_na.append(1)
                continue

            token_start_index = None
            token_end_index = None
            for i in con_tok_idx:
                sc, ec = offsets[i]
                if ec > s:
                    token_start_index = i
                    break
            for i in reversed(con_tok_idx):
                sc, ec = offsets[i]
                if sc < e:
                    token_end_index = i
                    break

            has_answer_here = (
                token_start_index is not None and
                token_end_index   is not None and
                token_start_index <= token_end_index
            )
            if not has_answer_here:
                continue

            labels = []
            started = False
            for i, sid in enumerate(seq_ids):
                if sid != 1:
                    labels.append(-100)
                else:
                    if i < token_start_index or i > token_end_index:
                        labels.append(label2id["O"])
                    else:
                        if not started:
                            labels.append(label2id["B-ANS"]); started = True
                        else:
                            labels.append(label2id["I-ANS"])
            all_input_ids.append(input_ids); all_attn.append(attn); all_labels.append(labels); all_na.append(0)
            kept_pos = True
            break

        if ans and (not kept_pos) and n_feats > 0:
            input_ids = enc["input_ids"][0]; attn = enc["attention_mask"][0]; seq_ids = enc.sequence_ids(0)
            labels = [(-100 if sid != 1 else label2id["O"]) for sid in seq_ids]
            all_input_ids.append(input_ids); all_attn.append(attn); all_labels.append(labels); all_na.append(0)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attn,
        "labels": all_labels,
        "na_label": all_na,
    })

class TokenPlusNoAnswer(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        drop = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop)
        self.token_classifier = nn.Linear(hidden, num_labels)
        self.na_classifier    = nn.Linear(hidden, 1)

    def forward(self, input_ids=None, attention_mask=None, labels=None, na_label=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        seq = self.dropout(out.last_hidden_state)
        token_logits = self.token_classifier(seq)
        cls_repr = seq[:, 0, :]
        na_logit = self.na_classifier(cls_repr)
        return {"logits": (token_logits, na_logit)}


def decode_with_na(token_logits: torch.Tensor, na_logits: torch.Tensor, threshold: float):
    na_prob = torch.sigmoid(na_logits).squeeze(-1)
    pred_tok = token_logits.argmax(-1) 
    force_empty = (na_prob >= threshold)
    if force_empty.any():
        pred_tok[force_empty] = label2id["O"] 
    return pred_tok.cpu().numpy()

def compute_metrics(p):
    tok_logits, na_logits = p.predictions
    tok_logits = torch.as_tensor(tok_logits)
    na_logits  = torch.as_tensor(na_logits).view(-1, 1)

    labels_field = p.label_ids
    if isinstance(labels_field, dict):
        labels = labels_field.get("labels")
    elif isinstance(labels_field, (list, tuple)):
        labels = labels_field[0] 
    else:
        labels = labels_field

    labels = np.asarray(labels)

    preds = decode_with_na(tok_logits, na_logits, no_answer_threshold)

    true_seqs, pred_seqs = [], []
    for y_true_row, y_pred_row in zip(labels, preds):
        y_true_row = np.asarray(y_true_row).reshape(-1)
        y_pred_row = np.asarray(y_pred_row).reshape(-1)

        t_tags, p_tags = [], []
        for t_id, p_id in zip(y_true_row, y_pred_row):
            t_id = int(t_id)
            if t_id == -100:
                continue
            p_id = int(p_id)
            t_tags.append(id2label[t_id])
            p_tags.append(id2label[p_id])

        true_seqs.append(t_tags)
        pred_seqs.append(p_tags)

    f1   = f1_score(true_seqs, pred_seqs, suffix=False)
    prec = precision_score(true_seqs, pred_seqs, suffix=False)
    rec  = recall_score(true_seqs, pred_seqs, suffix=False)
    exact = float(np.mean([t == p for t, p in zip(true_seqs, pred_seqs)]))
    return {"f1": f1, "precision": prec, "recall": rec, "exact_span_match": exact}

def preprocess_logits_for_metrics(logits, labels):
    token_logits, na_logits = logits
    return (token_logits.detach().to(torch.float32).cpu(),
            na_logits.detach().to(torch.float32).cpu())


class MultiTaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels   = inputs.pop("labels")
        na_label = inputs.pop("na_label")
        outputs  = model(**inputs)
        token_logits, na_logits = outputs["logits"]

        B, L, C = token_logits.shape
        weight = torch.tensor([1.0, 6.0, 3.0], device=token_logits.device)
        token_loss = F.cross_entropy(
            token_logits.view(B * L, C),
            labels.view(B * L),
            weight=weight, ignore_index=-100,
            label_smoothing=0.03
        )

        na_label = na_label.to(token_logits.device).float().view(-1, 1)
        na_loss = F.binary_cross_entropy_with_logits(na_logits, na_label)

        loss = token_loss + lambda_na * na_loss
        return (loss, outputs) if return_outputs else loss

model_name = "xlm-roberta-base"
max_len    = 512
doc_stride = 128
seed       = 42

label_list = ["O", "B-ANS", "I-ANS"]
id2label   = {i: l for i, l in enumerate(label_list)}
label2id   = {l: i for i, l in enumerate(label_list)}

no_answer_threshold = 0.50
lambda_na = 0.5


def run_language(lang_code: str):
    print(f"\n=== Span labeling — language: {lang_code} ===")

    tr_all = pd.read_csv(train_csv)
    va_all = pd.read_csv(val_csv)

    tr = tr_all[tr_all["lang"] == lang_code].copy()
    va = va_all[va_all["lang"] == lang_code].copy()

    for df in (tr, va):
        df["question"]   = df["question"].astype(str)
        df["context"]    = df["context"].astype(str)
        df["answer"]     = df["answer"].astype(str)
        df["answerable"] = df["answerable"].astype(bool)

    add_char_spans(tr); add_char_spans(va)
    span_coverage_report(tr, f"{lang_code}/train")
    span_coverage_report(va, f"{lang_code}/val")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = TokenPlusNoAnswer(model_name, num_labels=len(label_list))

    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()

    tr_ds = build_windows_dataset(tr, tokenizer)
    va_ds = build_windows_dataset(va, tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(filepath, f"span_lab_{lang_code}"),
        optim="adamw_torch",
        lr_scheduler_type="linear",
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=16,
        weight_decay=0.0,
        warmup_ratio=0.06,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        fp16=True,
        fp16_full_eval=True,
        eval_accumulation_steps=32,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = trainer.evaluate()
    print("\nValidation metrics:", json.dumps(metrics, indent=2, ensure_ascii=False))
    del metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir = os.path.join(filepath, f"span_lab_{lang_code}")
    trainer.args.save_safetensors = False
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[{lang_code}] Saved to: {out_dir}")

    @torch.no_grad()
    def preview_examples(df, n_ans=2, n_unans=2, thr=no_answer_threshold):
        model.eval()
        pick_ans   = df[df["answerable"] == True].head(n_ans)
        pick_unans = df[df["answerable"] == False].head(n_unans)

        def decode_best_spans(enc, token_logits, na_logits, thr):
            pred_w = decode_with_na(token_logits, na_logits, thr)[0]
            seq_ids = enc.sequence_ids(0)
            offsets = enc["offset_mapping"][0].tolist()
            spans, cur = [], None
            prev_is_ans = False
            for lab, sid, (a, b) in zip(pred_w, seq_ids, offsets):
                if sid != 1:
                    continue
                tag = id2label[int(lab)]
                if tag == "B-ANS" or (tag == "I-ANS" and not prev_is_ans):
                    if cur: spans.append(cur)
                    cur = [a, b]; prev_is_ans = True
                elif tag == "I-ANS" and cur:
                    cur[1] = b; prev_is_ans = True
                else:
                    if cur: spans.append(cur)
                    cur = None; prev_is_ans = False
            if cur: spans.append(cur)
            return spans

        for tag, subset in (("ANS", pick_ans), ("UNANS", pick_unans)):
            for i, ex in enumerate(subset.to_dict(orient="records"), 1):
                enc = tokenizer(
                    ex["question"], ex["context"],
                    truncation="only_second",
                    max_length=max_len,
                    stride=doc_stride,
                    return_overflowing_tokens=False,
                    return_offsets_mapping=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(
                        input_ids=enc["input_ids"].to(next(model.parameters()).device),
                        attention_mask=enc["attention_mask"].to(next(model.parameters()).device)
                    )
                tok_logit, na_logit = out["logits"]
                spans = decode_best_spans(enc, tok_logit.cpu(), na_logit.cpu(), thr)
                pred_texts = [ex["context"][a:b] for (a, b) in spans]
                print(f"\n[{tag}] Ex #{i} — lang={lang_code}")
                print("Q:", ex["question"])
                print("Pred:", "NO ANSWER" if len(pred_texts) == 0 else pred_texts)
                print("Gold:", ex["answer"] if tag=="ANS" else "<NO ANSWER>")
                del enc, out, tok_logit, na_logit
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    preview_examples(va, n_ans=5, n_unans=5, thr=no_answer_threshold)

    del trainer, model, tokenizer, tr_ds, va_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    for lang in ["ar", "ko", "te"]:
        run_language(lang)

# results (ar):
# Validation metrics: {
#   "eval_loss": 0.6207766532897949,
#   "eval_f1": 0.21024734982332155,
#   "eval_precision": 0.15474642392717816,
#   "eval_recall": 0.3278236914600551,
#   "eval_exact_span_match": 0.25783132530120484,
#   "eval_runtime": 2.5761,
#   "eval_samples_per_second": 161.095,
#   "eval_steps_per_second": 40.371,
#   "epoch": 16.0
# }
# [ar] Saved to: C:\Users\kkove\Desktop\NLP_project\span_lab_ar

# [ANS] Ex #1 — lang=ar
# Q: ما هي أولى جامعات فنلندا؟
# Pred: ['University', 'Helsinki']
# Gold: Royal Academy of Åbo

# [ANS] Ex #2 — lang=ar
# Q: ما عدد الدول المطلة على بحر البلطيق؟
# Pred: ['Finland, Sweden, Denmark, Estonia, Latvia, Lithuania, northwest Russia, Poland, Germany and the North and Central European Plain']
# Gold: Finland, Sweden, Denmark, Estonia, Latvia, Lithuania, northwest Russia, Poland, Germany

# [ANS] Ex #3 — lang=ar
# Q: اين عاش نيوتن؟
# Pred: ['to age', '17,', '16', 'eight']
# Gold: Grantham

# [ANS] Ex #4 — lang=ar
# Q: من هو الرئيس الأول للجمهورية اليمنية؟
# Pred: ['Abdrabbuh Mansur Hadi']
# Gold: Ali Abdullah Saleh

# [ANS] Ex #5 — lang=ar
# Q: متى إنتهت حرب نورماندي ؟
# Pred: ['1 May']
# Gold: 1944

# [UNANS] Ex #1 — lang=ar
# Q: هل زار ابن بطوطة اليمن؟
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #2 — lang=ar
# Q: هل توجد قواعد عسكرية فرنسية في جيبوتي؟
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #3 — lang=ar
# Q: هل خاضت مملكة اسبانيا حروب مع انجلترا؟
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #4 — lang=ar
# Q: هل يوجد لقاح للحدّ من مرض حمى الضنك؟
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #5 — lang=ar
# Q: هل هناك تساوي بين الرجل والمرأة في الهند ؟
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# results (ko):
# Validation metrics: {
#   "eval_loss": 0.7134074568748474,
#   "eval_f1": 0.3873517786561264,
#   "eval_precision": 0.34834123222748814,
#   "eval_recall": 0.4362017804154303,
#   "eval_exact_span_match": 0.398876404494382,
#   "eval_runtime": 1.9555,
#   "eval_samples_per_second": 182.048,
#   "eval_steps_per_second": 45.512,
#   "epoch": 16.0
# }
# [ko] Saved to: C:\Users\kkove\Desktop\NLP_project\span_lab_ko

# [ANS] Ex #1 — lang=ko
# Q: 북유럽의 노르딕 국가는 몇개인가요?
# Pred: ['12', 'four']
# Gold: five

# [ANS] Ex #2 — lang=ko
# Q: 1887년 케이스 웨스턴 리저브 대학의 이름은 무엇인가?
# Pred: ['Case Western Reserve University']
# Gold: Western Reserve University (formerly Western Reserve College) and Case Institute of Technology (formerly Case School of Applied Science)

# [ANS] Ex #3 — lang=ko
# Q: 옴진리교는 어느 나라에서 시작된 종교인가?
# Pred: ['Egypt', 'Greek']
# Gold: Egypt

# [ANS] Ex #4 — lang=ko
# Q: 댈러스의 면적은 얼마나 되나요?
# Pred: ['385.8 square miles (999.3 km2).']
# Gold: 999.3 km2

# [ANS] Ex #5 — lang=ko
# Q: 오픈스택의 프로그래밍 언어는 무엇인가요?
# Pred: ['Python']
# Gold: Python

# [UNANS] Ex #1 — lang=ko
# Q: 시차는 중력과 관련이 있는가?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #2 — lang=ko
# Q: 맹장 없이 살 수 있을까?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #3 — lang=ko
# Q: 무한은 공식이 있을까?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #4 — lang=ko
# Q: 핵융합은 상용화 가능성이 있는가?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #5 — lang=ko
# Q: 화성의 대기에 인간이 살 수 있는가?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# results (te):
# Validation metrics: {
#   "eval_loss": 1.027826189994812,
#   "eval_f1": 0.21846153846153843,
#   "eval_precision": 0.1977715877437326,
#   "eval_recall": 0.24398625429553264,
#   "eval_exact_span_match": 0.3072916666666667,
#   "eval_runtime": 2.1049,
#   "eval_samples_per_second": 182.436,
#   "eval_steps_per_second": 45.609,
#   "epoch": 16.0
# }
# [te] Saved to: C:\Users\kkove\Desktop\NLP_project\span_lab_te

# [ANS] Ex #1 — lang=te
# Q: ఒరెగాన్ రాష్ట్రంలోని అతిపెద్ద నగరం ఏది ?
# Pred: ['Seattle']
# Gold: Portland

# [ANS] Ex #2 — lang=te
# Q: కలరా వ్యాధిని మొదటగా ఏ దేశంలో కనుగొన్నారు ?
# Pred: ['1961']
# Gold: Indian subcontinent

# [ANS] Ex #3 — lang=te
# Q: కలరా వ్యాధిని మొదటగా ఏ దేశంలో కనుగొన్నారు ?
# Pred: NO ANSWER
# Gold: England

# [ANS] Ex #4 — lang=te
# Q: మొదటి ప్రపంచ యుద్ధం ఎప్పుడు మొదలయింది ?
# Pred: ['1914', '1918']
# Gold: 1914

# [ANS] Ex #5 — lang=te
# Q: మొదటి ప్రపంచ యుద్ధం ఎప్పుడు మొదలయింది ?
# Pred: NO ANSWER
# Gold: 28 July 1914

# [UNANS] Ex #1 — lang=te
# Q: మలేరియా వ్యాధి కి మందు కనిపెట్టిన శాస్త్రవేత్త ఎవరు?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #2 — lang=te
# Q: ఈస్ట్ ఇండియా కంపెనీ భారతదేశంలోకి ఎప్పుడు వచ్చింది?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #3 — lang=te
# Q: తెలుగు పంచాంగం ప్రకారం నూతన సంవత్సరం ఏ ఇంగ్లీష్ నెలలో ప్రారంభమవుతుంది?
# Pred: NO ANSWER
# Gold: <NO ANSWER>

# [UNANS] Ex #4 — lang=te
# Q: దక్షిణ ఆఫ్రికా దేశ విస్తీర్ణం ఎంత?
# Pred: ['6.9']
# Gold: <NO ANSWER>

# [UNANS] Ex #5 — lang=te
# Q: న్యూయార్క్ యొక్క జనసాంద్రత ఎంత ?
# Pred: ['304.8 square miles', '468.9']
# Gold: <NO ANSWER>