import pandas as pd
from datasets import Dataset as HFDataset

# Load CSV dataset
DATASET_PATH = "smoldoc_cleaned_parallel.csv"
df = pd.read_csv(DATASET_PATH)

# Display first few rows to understand structure
print(df.head())

# Assuming the dataset has two columns named 'kokborok' and 'english' for source and target languages
src_texts = df['kokborok'].astype(str).tolist()
tgt_texts = df['english'].astype(str).tolist()

# Convert to HuggingFace Dataset
dataset = HFDataset.from_dict({'src_text': src_texts, 'tgt_text': tgt_texts})

# Check dataset samples
print(dataset[0])
