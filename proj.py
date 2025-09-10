import pandas as pd

df_train = pd.read_csv("tydi_xor_rc_train.csv")
df_val = pd.read_csv("tydi_xor_rc_validation.csv")

langs = ["ar", "ko", "te"]

df_train_filtered = df_train[df_train["lang"].isin(langs)]
df_val_filtered   = df_val[df_val["lang"].isin(langs)]
