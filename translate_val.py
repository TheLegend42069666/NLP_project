import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import trange
import torch

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_val = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil.csv"))

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

# --- Translate questions in validation set and save ---
print("\nTranslating df_val questions to English and saving...")
df_out = df_val.copy()
df_out["question_en"] = ""

for i in langs:
    subset = df_out[df_out["lang"] == i]
    if subset.empty:
        continue
    questions = subset["question"].astype(str).tolist()
    que_eng = batch_translate(
        questions, src_lang=langs_nllb[i], tgt_lang=target_lang, batch_size=16
    )
    # write back in the same order
    df_out.loc[subset.index, "question_en"] = que_eng

out_path = os.path.join(filepath, "val_ar_ko_te_fil_tran.csv")
df_out.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved translated validation CSV to: {out_path}")
