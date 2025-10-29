import os
import json
import argparse
import torch
import numpy as np
from typing import List, Dict

from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    GenerationConfig
)
from sacrebleu.metrics import BLEU, CHRF

from pathlib import Path
filepath = Path(__file__).resolve().parents[1]


def safe_batch_decode(ids_like, tok):
    if isinstance(ids_like, tuple):
        ids_like = ids_like[0]

    if isinstance(ids_like, torch.Tensor):
        arr = ids_like.detach().cpu().numpy()
    else:
        arr = np.asarray(ids_like)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    vocab_size = getattr(tok, "vocab_size", None) or len(tok)
    arr = np.where((arr < 0) | (arr >= vocab_size), pad_id, arr).astype(np.int64)
    return tok.batch_decode(arr.tolist(), skip_special_tokens=True)

def row_to_src(que: str, con: str) -> str:
    return (
        "task: qa\n"
        "lang: te\n"
        "Answer in fluent Telugu using ONLY the context.\n"
        f"question: {que.strip()}\n"
        f"context: {con.strip()}\n"
    )

def load_test(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = [r for r in data if str(r.get("lang", "")).lower() == "te"]
    return data

def compute_scores(preds: List[str], refs: List[str]) -> Dict[str, float]:
    bleu1 = BLEU(tokenize="none", effective_order=True, max_ngram_order=1).corpus_score(preds, [refs]).score
    bleu2 = BLEU(tokenize="none", effective_order=True, max_ngram_order=2).corpus_score(preds, [refs]).score
    bleu3 = BLEU(tokenize="none", effective_order=True, max_ngram_order=3).corpus_score(preds, [refs]).score
    bleu4 = BLEU(tokenize="none", effective_order=True, max_ngram_order=4).corpus_score(preds, [refs]).score
    chrf  = CHRF().corpus_score(preds, [refs]).score
    denom = (sum(len(r.split()) for r in refs) + 1e-9)
    len_ratio = (sum(len(p.split()) for p in preds) + 1e-9) / denom
    return dict(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4, chrf=chrf, len_ratio=len_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default=os.path.join(filepath, "mt5base_te_qa_merged"))
    ap.add_argument("--test_json", type=str, default="test.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_src_len", type=int, default=1024)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    gen_cfg = GenerationConfig.from_pretrained(args.model_dir)

    rows = load_test(args.test_json)

    srcs, refs, ids, ques, cons, answerable = [], [], [], [], [], []
    for r in rows:
        ids.append(r.get("id", None))
        ques.append(str(r.get("question", "")))
        cons.append(str(r.get("context", "")))
        srcs.append(row_to_src(ques[-1], cons[-1]))
        refs.append(str(r.get("answer_inlang", "")).strip())
        answerable.append(bool(r.get("answerable", True)))

    preds: List[str] = []

    with torch.inference_mode():
        for i in range(0, len(srcs), args.batch_size):
            batch = srcs[i:i+args.batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=args.max_src_len,
                return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model.generate(**enc, generation_config=gen_cfg)
            batch_text = safe_batch_decode(out, tokenizer)
            preds.extend(batch_text)

    scores_all = compute_scores(preds, refs)
    print(f"\nOverall [{len(preds)}]: "
          f"BLEU1={scores_all['bleu1']:.2f} | BLEU2={scores_all['bleu2']:.2f} | "
          f"BLEU3={scores_all['bleu3']:.2f} | BLEU4={scores_all['bleu4']:.2f} | "
          f"chrF={scores_all['chrf']:.2f} | len-ratio={scores_all['len_ratio']:.2f}")

    import numpy as np
    ans_idx = [i for i, a in enumerate(answerable) if a]
    unans_idx = [i for i, a in enumerate(answerable) if not a]

    if ans_idx:
        a_preds = [preds[i] for i in ans_idx]
        a_refs  = [refs[i]  for i in ans_idx]
        a_scores = compute_scores(a_preds, a_refs)
        print(f"\nAnswerable [{len(a_preds)}]: "
              f"BLEU1={a_scores['bleu1']:.2f} | BLEU2={a_scores['bleu2']:.2f} | "
              f"BLEU3={a_scores['bleu3']:.2f} | BLEU4={a_scores['bleu4']:.2f} | "
              f"chrF={a_scores['chrf']:.2f} | len-ratio={a_scores['len_ratio']:.2f}")

    if unans_idx:
        u_preds = [preds[i] for i in unans_idx]
        u_refs  = [refs[i]  for i in unans_idx]
        u_scores = compute_scores(u_preds, u_refs)
        print(f"\nUnanswerable [{len(u_preds)}]: "
              f"BLEU1={u_scores['bleu1']:.2f} | BLEU2={u_scores['bleu2']:.2f} | "
              f"BLEU3={u_scores['bleu3']:.2f} | BLEU4={u_scores['bleu4']:.2f} | "
              f"chrF={u_scores['chrf']:.2f} | len-ratio={u_scores['len_ratio']:.2f}")

    print("\nSample predictions:")
    for i in range(len(rows)):
        print("----")
        print(f"id: {ids[i]} | answerable: {answerable[i]}")
        print("Q:", ques[i])
        print("PRED:", preds[i])
        print("REF: ", refs[i])

if __name__ == "__main__":
    main()
