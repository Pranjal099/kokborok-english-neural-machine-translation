import requests
import json
import pandas as pd
import re

url = "https://huggingface.co/datasets/sdmy/kokborok/raw/main/smoldoc__en_trp.json"
local_json_file = "smoldoc__en_trp.json"

try:
    print("Downloading dataset...")
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if download failed

    # Save JSON locally
    with open(local_json_file, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded and saved JSON to {local_json_file}")

    # Load JSON file
    with open(local_json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process as before
    rows = []

    def normalize(text):
        text = str(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    for topic in data:
        english_paragraph = normalize(" ".join(topic["srcs"]))
        kokborok_paragraph = normalize(" ".join(topic["trgs"]))
        if english_paragraph and kokborok_paragraph:
            rows.append({
                "english": english_paragraph,
                "kokborok": kokborok_paragraph
            })

    df = pd.DataFrame(rows)
    output_file = "smoldoc_cleaned_parallel.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} cleaned paragraph pairs to {output_file}")

except requests.exceptions.RequestException as e:
    print(f"Download failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
