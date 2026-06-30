# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# import time, csv, re, os, pandas as pd


# # Bible book codes and number of chapters (Old + New Testament)
# BOOKS = [
#     ("GEN", 50), ("EXO", 40), ("LEV", 27), ("NUM", 36), ("DEU", 34),
#     ("JOS", 24), ("JDG", 21), ("RUT", 4), ("1SA", 31), ("2SA", 24),
#     ("1KI", 22), ("2KI", 25), ("1CH", 29), ("2CH", 36), ("EZR", 10),
#     ("NEH", 13), ("EST", 10), ("JOB", 42), ("PSA", 150), ("PRO", 31),
#     ("ECC", 12), ("SNG", 8), ("ISA", 66), ("JER", 52), ("LAM", 5),
#     ("EZK", 48), ("DAN", 12), ("HOS", 14), ("JOL", 3), ("AMO", 9),
#     ("OBA", 1), ("JON", 4), ("MIC", 7), ("NAH", 3), ("HAB", 3),
#     ("ZEP", 3), ("HAG", 2), ("ZEC", 14), ("MAL", 4),
#     ("MAT", 28), ("MRK", 16), ("LUK", 24), ("JHN", 21), ("ACT", 28),
#     ("ROM", 16), ("1CO", 16), ("2CO", 13), ("GAL", 6), ("EPH", 6),
#     ("PHP", 4), ("COL", 4), ("1TH", 5), ("2TH", 3), ("1TI", 6),
#     ("2TI", 4), ("TIT", 3), ("PHM", 1), ("HEB", 13), ("JAS", 5),
#     ("1PE", 5), ("2PE", 3), ("1JN", 5), ("2JN", 1), ("3JN", 1),
#     ("JUD", 1), ("REV", 22)
# ]


# BASE_URL = "https://www.bible.com/bible/1540/{book}.{chap}.KOKBSI?parallel=1"
# OUT_CSV = "bible_parallel_full.csv"
# PROGRESS_FILE = "progress.txt"


# def clean_text(text):
#     if not text:
#         return ""
#     text = re.sub(r'^\s*\d+[\.\s]*', '', text)  # remove leading verse numbers
#     text = text.replace("“", "").replace("”", "").replace("’", "").replace("‘", "")
#     text = text.replace("‚", "").replace("‛", "").replace("´", "").replace("`", "")
#     text = text.replace("€™", "").replace("�", "")
#     return text.strip().lower()


# def scrape_chapter(book, chap, driver):
#     url = BASE_URL.format(book=book, chap=chap)
#     driver.get(url)
    
#     WebDriverWait(driver, 30).until(
#         EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-usfm]"))
#     )
    
#     last_height = 0
#     while True:
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(1.5)
#         new_height = driver.execute_script("return document.body.scrollHeight")
#         if new_height == last_height:
#             break
#         last_height = new_height
    
#     spans = driver.find_elements(By.CSS_SELECTOR, "span[data-usfm]")
    
#     vw = driver.execute_script("return window.innerWidth")
#     mid_x = vw / 2.0
    
#     left_spans, right_spans = [], []
#     for el in spans:
#         txt = el.text.strip()
#         if not txt:
#             continue
#         cx = el.location["x"] + el.size["width"] / 2
#         (left_spans if cx < mid_x else right_spans).append(el)
    
#     def collect_by_usfm(elems):
#         by_id = {}
#         order = []
#         for el in elems:
#             vid = el.get_attribute("data-usfm") or ""
#             t = el.text.strip()
#             if not vid or not t:
#                 continue
#             if vid not in by_id:
#                 by_id[vid] = []
#                 order.append(vid)
#             by_id[vid].append(t)
#         return order, {vid: clean_text(" ".join(parts)) for vid, parts in by_id.items()}
    
#     left_order, left_joined = collect_by_usfm(left_spans)
#     right_order, right_joined = collect_by_usfm(right_spans)
    
#     def eng_score(txt):
#         return sum(w in txt for w in [" the ", " and ", " of "])
    
#     left_score = eng_score(" ".join(left_joined.values()))
#     right_score = eng_score(" ".join(right_joined.values()))
    
#     if left_score > right_score:
#         eng, kok = left_joined, right_joined
#     else:
#         kok, eng = left_joined, right_joined
    
#     all_ids = sorted(set(list(kok.keys()) + list(eng.keys())))
    
#     rows = []
#     for vid in all_ids:
#         k = kok.get(vid, "")
#         e = eng.get(vid, "")
#         if k or e:
#             rows.append((k,e))
#     return rows


# def read_progress():
#     if os.path.exists(PROGRESS_FILE):
#         with open(PROGRESS_FILE, "r") as f:
#             line = f.read().strip()
#             if "," in line:
#                 book, chap = line.split(",")
#                 return book, int(chap)
#     return "NAH", 1  # default start point


# def save_progress(book, chap):
#     with open(PROGRESS_FILE, "w") as f:
#         f.write(f"{book},{chap}")


# def main():
#     if os.path.exists(OUT_CSV):
#         df = pd.read_csv(OUT_CSV, encoding="utf-8-sig")
#         all_rows = df.values.tolist()
#         print(f"▶ Resuming: previously scraped {len(all_rows)} verses.")
#     else:
#         all_rows = []
#         with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
#             csv.writer(f).writerow(["kokborok", "english"])
    
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
#     total_verses = len(all_rows)
#     RESUME_BOOK, RESUME_CHAP = read_progress()
#     resume_started = False

#     for book, n_chaps in BOOKS:
#         if not resume_started:
#             if book == RESUME_BOOK:
#                 resume_started = True
#             else:
#                 continue
        
#         print(f"📖 Scraping book: {book}")
#         for chap in range(1, n_chaps + 1):
#             if book == RESUME_BOOK and chap < RESUME_CHAP:
#                 continue
            
#             try:
#                 verses = scrape_chapter(book, chap, driver)
#                 if verses:
#                     all_rows.extend(verses)
#                     total_verses += len(verses)
#                     with open(OUT_CSV, "a", newline="", encoding="utf-8-sig") as f:
#                         csv.writer(f).writerows(verses)
#                     save_progress(book, chap)
#                     print(f"  ✅ {book} {chap} scraped, {len(verses)} verses | Total so far: {total_verses}")
#                 else:
#                     print(f"  ⚠️ {book} {chap} has no verses scraped")
#             except Exception as ex:
#                 print(f"  ❌ Error scraping {book} {chap}: {ex}")
    
#     driver.quit()
#     print(f"\n🎉 Scraping complete! Total verses saved: {total_verses} in {OUT_CSV}")


# if __name__ == "__main__":
#     main()
import pandas as pd

# Load the CSV file
df = pd.read_csv('bible_parallel_full.csv', encoding='utf-8-sig')

# List of Bible books with chapter counts for reference
BOOKS = [
    ("GEN", 50), ("EXO", 40), ("LEV", 27), ("NUM", 36), ("DEU", 34),
    ("JOS", 24), ("JDG", 21), ("RUT", 4), ("1SA", 31), ("2SA", 24),
    ("1KI", 22), ("2KI", 25), ("1CH", 29), ("2CH", 36), ("EZR", 10),
    ("NEH", 13), ("EST", 10), ("JOB", 42), ("PSA", 150), ("PRO", 31),
    ("ECC", 12), ("SNG", 8), ("ISA", 66), ("JER", 52), ("LAM", 5),
    ("EZK", 48), ("DAN", 12), ("HOS", 14), ("JOL", 3), ("AMO", 9),
    ("OBA", 1), ("JON", 4), ("MIC", 7), ("NAH", 3), ("HAB", 3),
    ("ZEP", 3), ("HAG", 2), ("ZEC", 14), ("MAL", 4),
    ("MAT", 28), ("MRK", 16), ("LUK", 24), ("JHN", 21), ("ACT", 28),
    ("ROM", 16), ("1CO", 16), ("2CO", 13), ("GAL", 6), ("EPH", 6),
    ("PHP", 4), ("COL", 4), ("1TH", 5), ("2TH", 3), ("1TI", 6),
    ("2TI", 4), ("TIT", 3), ("PHM", 1), ("HEB", 13), ("JAS", 5),
    ("1PE", 5), ("2PE", 3), ("1JN", 5), ("2JN", 1), ("3JN", 1),
    ("JUD", 1), ("REV", 22)
]

# Since your CSV likely doesn't have book/chapter columns, you need to parse them.
# To do that, we'd need the dataset to include metadata for book/chapter per row.
# If you have such columns, replace below accordingly.

# If not, and if your scraping script collected book and chapter for each verse,
# then you can load them here. Otherwise, it's difficult to summarize by book/chapter.

print("If your CSV has columns for book and chapter, please share the column names so I can help generate the summary.")

# If you want, I can help you modify your scraping script to save book and chapter per verse for better summaries.

