"""Fetch arXiv submission dates for the 1,000-paper corpus.

Queries the arXiv API by paper title, extracts the published date,
and saves results to data/paper_submission_dates.csv.

Usage:
    python src/fetch_arxiv_dates.py [--batch-size 50] [--sleep 3.0]
"""

import argparse
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
GT_BATCH_INPUT = ROOT / "batches" / "ready_for_classification" / "1000batch_api.jsonl"
OUT_PATH = ROOT / "data" / "paper_submission_dates.csv"

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def parse_paper_titles(path: Path) -> dict[str, str]:
    titles = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            input_text = obj["body"].get("input", "")
            title_start = input_text.find("Title: ")
            if title_start >= 0:
                rest = input_text[title_start + 7:]
                title_end = rest.find("\n")
                title = rest[:title_end].strip() if title_end >= 0 else rest.strip()
                titles[custom_id] = title
    return titles


def query_arxiv(title: str) -> dict | None:
    query = f'ti:"{title}"'
    params = urllib.parse.urlencode({"search_query": query, "max_results": 1})
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        log.warning(f"Failed to fetch: {e}")
        return None

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", NS)
    if not entries:
        return None

    entry = entries[0]
    published = entry.findtext("atom:published", default="", namespaces=NS)
    arxiv_id_raw = entry.findtext("atom:id", default="", namespaces=NS)
    arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
    found_title = entry.findtext("atom:title", default="", namespaces=NS).strip()
    found_title = " ".join(found_title.split())

    return {
        "arxiv_id": arxiv_id,
        "published": published[:10] if published else "",
        "found_title": found_title,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch arXiv dates for corpus papers")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args()

    log.info("Resolved config:")
    log.info(f"  batch_size = {args.batch_size}")
    log.info(f"  sleep = {args.sleep}")

    titles = parse_paper_titles(GT_BATCH_INPUT)
    log.info(f"Loaded {len(titles)} paper titles")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if OUT_PATH.exists():
        df_existing = pd.read_csv(OUT_PATH)
        existing = {row["paper_id"]: row for _, row in df_existing.iterrows()}
        log.info(f"Resuming: {len(existing)} papers already fetched")

    results = []
    to_fetch = [(pid, title) for pid, title in sorted(titles.items()) if pid not in existing]
    log.info(f"Papers to fetch: {len(to_fetch)}")

    for i, (pid, title) in enumerate(to_fetch):
        if i > 0 and i % args.batch_size == 0:
            log.info(f"  Sleeping {args.sleep}s (arXiv rate limit)...")
            time.sleep(args.sleep)

        result = query_arxiv(title)
        if result:
            results.append({
                "paper_id": pid,
                "title": title,
                "arxiv_id": result["arxiv_id"],
                "published": result["published"],
                "found_title": result["found_title"],
            })
        else:
            results.append({
                "paper_id": pid,
                "title": title,
                "arxiv_id": "",
                "published": "",
                "found_title": "",
            })

        if (i + 1) % 50 == 0:
            log.info(f"  Fetched {i + 1}/{len(to_fetch)}")

    all_results = list(existing.values()) + results
    df = pd.DataFrame(all_results)
    df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved {len(df)} results to {OUT_PATH}")

    if "published" in df.columns:
        df["published"] = pd.to_datetime(df["published"], errors="coerce")
        valid = df[df["published"].notna()]
        log.info(f"Valid dates: {len(valid)}/{len(df)}")
        if not valid.empty:
            log.info(f"Date range: {valid['published'].min()} to {valid['published'].max()}")
            log.info(f"By month:")
            for month, count in valid["published"].dt.to_period("M").value_counts().sort_index().items():
                log.info(f"  {month}: {count}")


if __name__ == "__main__":
    main()
