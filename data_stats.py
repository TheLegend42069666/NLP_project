import pandas as pd
from collections import Counter
# import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import regex as re

filepath = "C:/Users/kkove/Desktop/NLP_project/"

df_train = pd.read_csv(filepath+"train_ar_ko_te_fil.csv")
df_val = pd.read_csv(filepath+"val_ar_ko_te_fil.csv")

langs = ["ar", "ko", "te"]

print("Traning set Size")
train_counts = {}
train_total_rows = df_train.shape[0]
print(f"Total amount of rows: {train_total_rows}")
print("Amount of rows per language:")
for i in langs:
    train_counts[i] = df_train[df_train["lang"] == i].shape[0]
print(train_counts)

print("\nValidation set Size")
val_counts = {}
val_total_rows = df_val.shape[0]
print(f"Total amount of rows: {val_total_rows}")
print("Amount of rows per language:")
for i in langs:
    val_counts[i] = df_val[df_val["lang"] == i].shape[0]
print(val_counts)

print("\nPercent of answerable questions:")
train_percent_answerable = df_train[df_train["answerable"]].shape[0]/df_train.shape[0]*100
print(f"Training set: {train_percent_answerable}%")

val_percent_answerable = df_val[df_val["answerable"]].shape[0]/df_val.shape[0]*100
print(f"Validation set: {val_percent_answerable}%")

#https://medium.com/@FaridSharaf/text-translation-using-nllb-and-huggingface-tutorial-7e789e0f7816
model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

target_lang = "eng_Latn"

langs_nllb = {"ar": "arb_Arab", "ko": "kor_Hang", "te": "tel_Telu"}

print("\nTotal amount of words for each language:")
translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    max_length=400
)

for i in langs:
    match i:
        case "ar":
            lang = "Arabic"
        case "ko":
            lang = "Korean"
        case "te":
            lang = "Telugu"
    text = " ".join((df_train[df_train["lang"]==i])["question"].astype(str))
    text = re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", text)
    tokens = text.split()
    unique_tokens = set(tokens)
    print(f"{lang}: {len(tokens)} ({len(unique_tokens)} unique)")

    top5_words = Counter(tokens).most_common(5)
    translations = translator(
        [w for w, _ in top5_words],
        src_lang=langs_nllb[i],
        tgt_lang=target_lang
    )
    print(f"Top 5 most occuring words {lang}:")
    for idx, ((word, count), trans) in enumerate(zip(top5_words, translations), start=1):
        print(f"{idx}. {word} ({count}), eng: {trans['translation_text']}")
    print("\n")