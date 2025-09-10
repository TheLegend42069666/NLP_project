from datasets import load_dataset

dataset = load_dataset("coastalcph/tydi_xor_rc")
df_train = dataset["train"].to_pandas()
df_val = dataset["validation"].to_pandas()

filepath = "C:/Users/kkove/Desktop/NLP_project"

df_train.to_csv(filepath+"tydi_xor_rc_train.csv", index=False)
df_val.to_csv(filepath+"tydi_xor_rc_validation.csv", index=False)

print("downloaded")