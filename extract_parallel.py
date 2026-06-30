import easyocr
import pandas as pd
import os
import re

reader = easyocr.Reader(['en'])

image_folder = "manual_trans"

# List all image files in folder (jpg, png, jpeg)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()

pairs = []

def clean_text(text):
    return ' '.join(text.split())

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    print(f"Processing {img_path} ...")
    lines = reader.readtext(img_path, detail=0, paragraph=False)

    for line in lines:
        # Try splitting by comma
        if ',' in line:
            eng, kok = [clean_text(s) for s in line.split(',', 1)]
        else:
            # Split on two or more spaces or tab (adjust if needed)
            parts = re.split(r'\s{2,}|\t', line)
            if len(parts) == 2:
                eng, kok = [clean_text(s) for s in parts]
            else:
                continue

        if eng and kok:
            pairs.append({'english': eng, 'kokborok': kok})

df = pd.DataFrame(pairs)
df.drop_duplicates(inplace=True)
df = df[(df['english'] != "") & (df['kokborok'] != "")]
df.reset_index(drop=True, inplace=True)

output_file = "learn_kokborok_parallel_clean.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved {len(df)} pairs to {output_file}")
