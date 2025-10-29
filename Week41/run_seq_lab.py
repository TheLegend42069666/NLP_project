import os, json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]
model_dir  = os.path.join(filepath, "span_lab_te")
input_json = os.path.join(filepath, "test.json")

model_name = "xlm-roberta-base"
max_len    = 512
doc_stride = 128
na_thr     = 0.50

label_list = ["O", "B-ANS", "I-ANS"]
id2label   = {i: l for i, l in enumerate(label_list)}
label2id   = {l: i for i, l in enumerate(label_list)}

class TokenPlusNoAnswer(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden = self.encoder.config.hidden_size
        drop = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop)
        self.token_classifier = nn.Linear(hidden, num_labels)
        self.na_classifier    = nn.Linear(hidden, 1)

    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.dropout(out.last_hidden_state)
        token_logits = self.token_classifier(seq)   
        cls_repr = seq[:, 0, :]                   
        na_logit = self.na_classifier(cls_repr)
        return token_logits, na_logit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

model = TokenPlusNoAnswer(model_name, num_labels=len(label_list))
state_path = os.path.join(model_dir, "pytorch_model.bin")
state = torch.load(state_path, map_location="cpu")
model.load_state_dict(state, strict=True)
model.to(device).eval()

def spans_from_preds(seq_ids, offsets, pred_ids):
    spans, cur = [], None
    prev_is_ans = False
    for lab_id, sid, (a, b) in zip(pred_ids, seq_ids, offsets):
        if sid != 1:
            continue
        tag = id2label.get(int(lab_id), "O")
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

@torch.no_grad()
def predict_one(que: str, con: str):
    enc = tokenizer(
        que, con,
        truncation="only_second",
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
        return_tensors=None
    )
    if len(enc["input_ids"]) == 0:
        return "NO ANSWER", True

    any_window_non_na = False
    for w in range(len(enc["input_ids"])):
        input_ids = torch.tensor([enc["input_ids"][w]], device=device)
        attn      = torch.tensor([enc["attention_mask"][w]], device=device)
        seq_ids   = enc.sequence_ids(w)

        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                tok_logits, na_logit = model(input_ids=input_ids, attention_mask=attn)
        else:
            tok_logits, na_logit = model(input_ids=input_ids, attention_mask=attn)

        na_prob = torch.sigmoid(na_logit).squeeze(-1)[0].item()
        pred_ids = tok_logits.argmax(-1)[0].tolist()

        if na_prob >= na_thr:
            continue

        any_window_non_na = True
        spans = spans_from_preds(seq_ids, enc["offset_mapping"][w], pred_ids)
        texts = []
        for a, b in spans:
            if a is None or b is None or a < 0 or b <= a or b > len(con):
                continue
            texts.append(con[a:b])

        if texts:
            return texts, False

    return "NO ANSWER", (not any_window_non_na)

def find_gold_window(enc, s_char: int, e_char: int):
    for w in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(w)
        offsets = enc["offset_mapping"][w]
        con_idx = [i for i, sid in enumerate(seq_ids) if sid == 1]
        if not con_idx:
            continue
        left  = any(offsets[i][1] > s_char for i in con_idx)
        right = any(offsets[i][0] < e_char for i in con_idx)
        if left and right:
            return w
    return None

def gold_tags_for_window(enc, w, s_char: int, e_char: int, answerable: bool):
    seq_ids = enc.sequence_ids(w)
    offsets = enc["offset_mapping"][w]
    tags = []
    if (not answerable) or s_char < 0 or e_char <= s_char:
        for i, sid in enumerate(seq_ids):
            if sid == 1:
                tags.append("O")
        return tags

    con_positions = [i for i, sid in enumerate(seq_ids) if sid == 1]
    span_token_idx = []
    for i in con_positions:
        a, b = offsets[i]
        if b > s_char and a < e_char: 
            span_token_idx.append(i)

    if not span_token_idx:
        for _ in con_positions:
            tags.append("O")
        return tags

    started = False
    for i in con_positions:
        if i in span_token_idx:
            tags.append("B-ANS" if not started else "I-ANS")
            started = True
        else:
            tags.append("O")
    return tags

@torch.no_grad()
def pred_tags_for_window(enc, w):
    input_ids = torch.tensor([enc["input_ids"][w]], device=device)
    attn      = torch.tensor([enc["attention_mask"][w]], device=device)
    seq_ids   = enc.sequence_ids(w)

    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            tok_logits, na_logit = model(input_ids=input_ids, attention_mask=attn)
    else:
        tok_logits, na_logit = model(input_ids=input_ids, attention_mask=attn)

    na_prob = torch.sigmoid(na_logit).squeeze(-1)[0].item()
    pred_ids = tok_logits.argmax(-1)[0].tolist()

    tags = []
    if na_prob >= na_thr:
        for i, sid in enumerate(seq_ids):
            if sid == 1:
                tags.append("O")
        return tags, True

    for i, sid in enumerate(seq_ids):
        if sid == 1:
            tags.append(id2label.get(int(pred_ids[i]), "O"))
    return tags, False

def load_tests(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    return data

def main():
    tests = load_tests(input_json)

    true_seqs, pred_seqs = [], []
    exact_flags = []
    na_total = 0
    na_correct = 0

    for i, ex in enumerate(tests, 1):
        que = str(ex.get("question", ""))
        con = str(ex.get("context", ""))
        gold_ans = str(ex.get("answer", ""))
        is_ans = bool(ex.get("answerable", False))

        pred_print, _ = predict_one(que, con)

        print(f"\nQ: {que}")
        if isinstance(pred_print, str) and pred_print == "NO ANSWER":
            print("Pred: NO ANSWER")
        else:
            print("Pred:", "[" + ", ".join([repr(t) for t in pred_print]) + "]")
        if is_ans:
            print(f"Gold: {gold_ans}")
        else:
            print("Gold: <NO ANSWER>")

        enc = tokenizer(
            que, con,
            truncation="only_second",
            max_length=max_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            return_tensors=None
        )
        if len(enc["input_ids"]) == 0:
            continue

        s_char = int(ex.get("answer_start", -1))
        e_char = s_char + len(gold_ans) if (s_char is not None and s_char >= 0) else -1
        if is_ans:
            w = find_gold_window(enc, s_char, e_char)
            if w is None:
                w = 0 
        else:
            w = 0
            na_total += 1

        gold_tags = gold_tags_for_window(enc, w, s_char, e_char, is_ans)
        pred_tags, forced_na = pred_tags_for_window(enc, w)

        if not is_ans:
            if forced_na or all(tag == "O" for tag in pred_tags):
                na_correct += 1

        true_seqs.append(gold_tags)
        pred_seqs.append(pred_tags)
        exact_flags.append(float(gold_tags == pred_tags))

    if len(true_seqs) == 0:
        print("\nValidation metrics: {}")
        return

    eval_f1 = f1_score(true_seqs, pred_seqs, suffix=False)
    eval_precision = precision_score(true_seqs, pred_seqs, suffix=False)
    eval_recall = recall_score(true_seqs, pred_seqs, suffix=False)
    eval_exact = float(np.mean(exact_flags))
    out = {
        "eval_f1": eval_f1,
        "eval_precision": eval_precision,
        "eval_recall": eval_recall,
        "eval_exact_span_match": eval_exact,
    }
    if na_total > 0:
        out["eval_na_accuracy"] = na_correct / na_total

    print("\nValidation metrics:", json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
