import pandas as pd
import os

base_path = "/Users/pranjalgautam/Documents/vs2/HELLOWORLD/Sahil.py"

files = [
    os.path.join(base_path, "bible_parallel_clean_norm.csv"),
    os.path.join(base_path, "learn_kokborok_parallel_clean.csv"),
    os.path.join(base_path, "smoldoc_clean_parallel.csv"),
    os.path.join(base_path, "smoldoc_cleaned_parallel.csv")
]

dfs = []
for f in files:
    try:
        print(f"🔹 Reading: {os.path.basename(f)}")
        df = pd.read_csv(f, on_bad_lines='skip', encoding='utf-8')  # 👈 Skip broken lines safely
        dfs.append(df)
        print(f"✅ Loaded {len(df)} rows")
    except Exception as e:
        print(f"⚠️ Error reading {f}: {e}")

# Combine datasets
df = pd.concat(dfs, ignore_index=True)

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]
if 'english' not in df.columns or 'kokborok' not in df.columns:
    raise ValueError("❌ Columns must be named 'english' and 'kokborok'")

# Drop missing or duplicate rows
df.dropna(subset=['english', 'kokborok'], inplace=True)
df.drop_duplicates(inplace=True)

# Keep only full sentences (more than 2 words)
def is_full_sentence(text):
    return len(str(text).split()) > 2

filtered_df = df[df['english'].apply(is_full_sentence) & df['kokborok'].apply(is_full_sentence)]

# Save final dataset
output_path = os.path.join(base_path, "kokborok_full_sentences.csv")
filtered_df.to_csv(output_path, index=False)

print("\n✅ Final dataset created successfully!")
print(f"📁 Saved to: {output_path}")
print(f"💬 Total sentence pairs retained: {len(filtered_df)}")
