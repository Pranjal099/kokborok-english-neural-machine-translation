import pandas as pd
import re
df = pd.read_csv('kokborok_dataset (1).csv', encoding='utf-8')



raw = df[df.columns[0]].astype(str)
entries = raw.str.split('entrpTrue').tolist()
all_pairs = [x for sublist in entries for x in sublist if x.strip()]
cleaned = []
for pair in all_pairs:
    m = re.match(r'^([a-zA-Z\\s,\']+)(.*)$', pair.strip())
    if m:
        english = m.group(1).strip()
        kokborok = m.group(2).strip()
        if english and kokborok:
            cleaned.append({'english': english, 'kokborok': kokborok})
cleaned_df = pd.DataFrame(cleaned)
cleaned_df.to_csv('kokborok_english_cleaned.csv', index=False)
print(cleaned_df.head())
