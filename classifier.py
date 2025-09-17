import os
import pandas as pd
import regex as re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

filepath = r"C:/Users/kkove/Desktop/NLP_project"

df_train = pd.read_csv(os.path.join(filepath, "train_ar_ko_te_fil_tran.csv"))
df_val   = pd.read_csv(os.path.join(filepath, "val_ar_ko_te_fil_tran.csv"))

langs = ["ar", "ko", "te"]

stopwords = set(stopwords.words("english"))

def normalize(text):
    return re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", str(text).lower())

def content_words(text):
    tokens = normalize(text).split()
    return [t for t in tokens if t not in stopwords]

def make_text(df):
    return (df["question_en"].astype(str) + " [SEP] " + df["context"].astype(str)).tolist()

print("\nLearned classifier (TF-IDF + Logistic Regression), eval on validation:")
for i in langs:
    train_subset = df_train[df_train["lang"] == i]
    val_subset   = df_val[df_val["lang"] == i]

    X_train = make_text(train_subset)
    y_train = train_subset["answerable"].astype(int).values
    X_val = make_text(val_subset)
    y_val = val_subset["answerable"].astype(int).values

    # Use YOUR tokenizer in TF-IDF (so we keep normalize+stopwords behavior)
    ranges = [(1,2), (1,3), (1, 4), (1,5)]
    best_range, best_acc = None, -1.0
    for range in ranges:

        vec = TfidfVectorizer(
            tokenizer=content_words,
            preprocessor=None,
            token_pattern=None, 
            lowercase=False, 
            ngram_range=(1, 2),         
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)

        classifier = LogisticRegression(
            penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=10000
        )
        classifier.fit(X_train_vec, y_train)
        acc  = accuracy_score(y_val, classifier.predict(X_val_vec))
        if acc > best_acc:
            best_acc = acc
            best_range = range
    print(f"\nBest ngram range: {best_range}, val accuracy: {best_acc:.3f}")
    y_predict = classifier.predict(X_val_vec)
    acc  = accuracy_score(y_val, y_predict)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, y_predict, average="binary", zero_division=0
    )

    print(f"Language: {i}")
    print(f"VAL samples: {len(y_val)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (answerable): {prec:.3f}")
    print(f"Recall (answerable): {rec:.3f}")
    print(f"F1 (answerable): {f1:.3f}")
