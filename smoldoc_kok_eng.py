import requests
import json
import pandas as pd
import re

# Download the JSON dataset
url = "https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json"
response = requests.get(url)
data = response.json()

# Flatten all sentence pairs
rows = []
for topic in data:
    srcs = topic["srcs"]
    trgs = topic["trgs"]
    for src, trg in zip(srcs, trgs):
        rows.append({"english": src, "kokborok": trg})

# Clean and normalize the sentences
def normalize(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = text.strip()
    return text

df = pd.DataFrame(rows)
df["english"] = df["english"].apply(normalize)
df["kokborok"] = df["kokborok"].apply(normalize)

# Remove empty rows if present
df = df[(df["english"] != "") & (df["kokborok"] != "")]

# Save to CSV
df.to_csv("smoldoc_clean_parallel.csv", index=False, encoding="utf-8-sig")

print(f"Saved {len(df)} cleaned aligned sentence pairs to 'smoldoc_clean_parallel.csv'")
