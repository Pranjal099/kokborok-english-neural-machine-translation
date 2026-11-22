"""Bilingual parallel corpus scraper for Kokborok–English Bible text.

Scrapes aligned verse pairs from the Kokborok Bible Society of India (KOKBSI)
with English parallel text on bible.com.
"""

import argparse
import csv
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Complete Bible canonical books (Old and New Testament) with chapter counts
BOOKS: List[Tuple[str, int]] = [
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

BASE_URL = "https://www.bible.com/bible/1540/{book}.{chap}.KOKBSI?parallel=1"


def clean_verse_text(text: str) -> str:
    """Normalize extracted verse text and remove artifact numbers and quotes."""
    if not text:
        return ""
    text = re.sub(r"^\s*\d+[\.\s]*", "", text)
    for quote in ["“", "”", "’", "‘", "‚", "‛", "´", "`", "€™", ""]:
        text = text.replace(quote, "")
    return " ".join(text.split()).strip().lower()


def scrape_chapter_verses(book: str, chap: int, driver) -> List[Tuple[str, str]]:
    """Scrape aligned verse pairs from a single chapter URL.

    Args:
        book: Book code (e.g. 'GEN').
        chap: Chapter index.
        driver: Active Selenium WebDriver.

    Returns:
        List of (kokborok_verse, english_verse) tuples.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    url = BASE_URL.format(book=book, chap=chap)
    driver.get(url)

    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "span[data-usfm]"))
    )

    # Scroll page to ensure dynamic spans load completely
    last_height = 0
    for _ in range(5):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.0)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    spans = driver.find_elements(By.CSS_SELECTOR, "span[data-usfm]")
    vw = driver.execute_script("return window.innerWidth")
    mid_x = vw / 2.0

    left_spans, right_spans = [], []
    for el in spans:
        txt = el.text.strip()
        if not txt:
            continue
        cx = el.location["x"] + el.size["width"] / 2.0
        (left_spans if cx < mid_x else right_spans).append(el)

    def collect_by_usfm(elems):
        by_id: Dict[str, List[str]] = {}
        for el in elems:
            vid = el.get_attribute("data-usfm") or ""
            t = el.text.strip()
            if not vid or not t:
                continue
            if vid not in by_id:
                by_id[vid] = []
            by_id[vid].append(t)
        return {vid: clean_verse_text(" ".join(parts)) for vid, parts in by_id.items()}

    left_dict = collect_by_usfm(left_spans)
    right_dict = collect_by_usfm(right_spans)

    def eng_score(d):
        joined = " ".join(d.values())
        return sum(joined.count(w) for w in [" the ", " and ", " of ", " in "])

    if eng_score(left_dict) > eng_score(right_dict):
        eng, kok = left_dict, right_dict
    else:
        kok, eng = left_dict, right_dict

    all_ids = sorted(set(list(kok.keys()) + list(eng.keys())))
    rows = []
    for vid in all_ids:
        k = kok.get(vid, "")
        e = eng.get(vid, "")
        if k and e:
            rows.append((k, e))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape bilingual Kokborok-English Bible text.")
    parser.add_argument("--output", type=str, default="bible_parallel_full.csv", help="Output CSV path.")
    parser.add_argument("--progress-file", type=str, default="progress.txt", help="File to persist progress.")
    parser.add_argument("--start-book", type=str, default="GEN", help="Starting book code.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Scraper configured. Note: Requires Chrome and selenium/webdriver-manager.")
