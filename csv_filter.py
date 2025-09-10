import pandas as pd

filepath = "C:/Users/kkove/Desktop/NLP_project/"

df_train = pd.read_csv(filepath+"tydi_xor_rc_train.csv")
df_val = pd.read_csv(filepath+"tydi_xor_rc_validation.csv")

langs = ["ar", "ko", "te"]

df_train_filtered = df_train[df_train["lang"].isin(langs)]
df_val_filtered   = df_val[df_val["lang"].isin(langs)]

df_train_filtered.to_csv(filepath+"train_ar_ko_te_fil.csv", index=False)
df_val_filtered.to_csv(filepath+"val_ar_ko_te_fil.csv", index=False)

print("filtered")