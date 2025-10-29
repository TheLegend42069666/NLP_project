import regex as re

text = "ప్రపంచంలో  మొట్టమొదటి దూర విద్య విద్యాలయం ఏ దేశంలో స్థాపించబడింది ? 30년 전쟁의 승자는 누구인가? متى تدخلت روسيا في  الحرب الأهلية السورية؟"

# print(re.sub(r"[^\p{Telugu}\p{Hangul}\p{Arabic}\p{N}\s]", " ", text))

text = re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", text)
# text = re.sub(r"\s+", " ", text).strip()
print(text)