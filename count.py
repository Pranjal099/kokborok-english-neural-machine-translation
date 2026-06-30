import pandas as pd

# Load the cleaned normalized CSV
df = pd.read_csv('bible_parallel_clean_norm.csv', encoding='utf-8-sig')

# Check for rows where Kokborok or English are empty or NaN
empty_kok = df['kokborok'].isna().sum() + (df['kokborok'].str.strip() == '').sum()
empty_eng = df['english'].isna().sum() + (df['english'].str.strip() == '').sum()

print(f"Empty Kokborok rows: {empty_kok}")
print(f"Empty English rows: {empty_eng}")
