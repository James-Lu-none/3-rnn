import pandas as pd
from pathlib import Path
import unicodedata
import re

input1 = Path("data/csv_output/詞目.csv")
meta1 = Path("data/train/train/dict-word.csv")
input2 = Path("data/csv_output/例句.csv")
meta2 = Path("data/train/train/dict-sentence.csv")

def clean_text(text):
    text = str(text)

    # (3) Keep only before '/'
    text = text.split('/')[0]

    # (2) Replace '-' with space
    text = text.replace('-', ' ')

    # (1) Remove tone marks
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    # (4) Remove non-alphabet characters (keep spaces)
    text = re.sub(r"[^A-Za-z\s]", "", text)

    # (5) Convert to lowercase
    text = text.lower().strip()

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text

df = pd.read_csv(input1)
df = df[['羅馬字', '羅馬字音檔檔名']].rename(columns={
    '羅馬字音檔檔名': 'id',
    '羅馬字': 'text'
}).dropna()
df["text"] = df["text"].astype(str).apply(clean_text)
df.to_csv(meta1, index=False)

df = pd.read_csv(input2)
df = df[['羅馬字', '音檔檔名']].rename(columns={
    '音檔檔名': 'id',
    '羅馬字': 'text'
}).dropna()
df["text"] = df["text"].astype(str).apply(clean_text)
df.to_csv(meta2, index=False)

