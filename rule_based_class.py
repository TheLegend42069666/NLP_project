import pandas as pd
import regex as re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from nltk.corpus import stopwords

filepath = "C:/Users/kkove/Desktop/NLP_project/"

df_train = pd.read_csv(filepath+"train_ar_ko_te_fil.csv")
df_val = pd.read_csv(filepath+"val_ar_ko_te_fil.csv")

langs = ["ar", "ko", "te"]

model_name = "facebook/nllb-200-distilled-600M"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

target_lang = "eng_Latn"
langs_nllb = {"ar": "arb_Arab", "ko": "kor_Hang", "te": "tel_Telu"}

translator = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    max_length=400
)

# # --- Simple English stopwords ---
# stopwords = {
#     "the","a","an","and","or","of","in","on","to","for","from","by","with",
#     "is","are","was","were","be","been","being",
#     "this","that","these","those","it","its","as","at","about","into","over","under",
#     "do","does","did","done","doing","can","could","may","might","should","would","will","shall",
#     "you","your","yours","me","my","mine","we","our","ours","they","their","theirs","he","she","his","her","hers","them","us"
# }

stopwords = set(stopwords.words("english"))

def normalize(text):
    return re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", str(text).lower())

def content_words(text):
    tokens = normalize(text).split()
    return [t for t in tokens if t not in stopwords]

def classify_answerable(question, context, thresh):
    que_words = content_words(question)
    con_words = set(content_words(context))
    overlap = sum(1 for w in que_words if w in con_words)
    ratio = overlap / max(1, len(que_words))
    # simple threshold
    if overlap >= thresh or ratio > 0.3:
        return True
    else:
        return False

# --- Evaluate on validation set ---
print("\nRule-based classifier performance:")
for i in langs:
    subset = df_val[df_val["lang"] == i]

    # translate questions to English
    que_translations = translator(
        subset["question"].astype(str).tolist(),
        src_lang=langs_nllb[i],
        tgt_lang=target_lang
    )
    q_eng = [t["translation_text"] for t in que_translations]

    preds = []
    corr_labels = subset["answerable"].tolist()

    for que, con in zip(q_eng, subset["context"].astype(str)):
        preds.append(classify_answerable(que, con, 2))

    tp = sum(pred and lab for pred, lab in zip(preds, corr_labels))
    tn = sum((not pred) and (not lab) for pred, lab in zip(preds, corr_labels))
    fp = sum(pred and (not lab) for pred, lab in zip(preds, corr_labels))
    fn = sum((not pred) and lab for pred, lab in zip(preds, corr_labels))

    acc = (tp+tn) / len(corr_labels)
    prec = tp / (tp+fp) if (tp+fp) else 0
    rec = tp / (tp+fn) if (tp+fn) else 0
    f1 = 2*prec*rec / (prec+rec) if (prec+rec) else 0

    print(f"\nLanguage: {i}")
    print(f"Total samples: {len(corr_labels)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (answerable): {prec:.3f}")
    print(f"Recall (answerable): {rec:.3f}")
    print(f"F1 (answerable): {f1:.3f}")
