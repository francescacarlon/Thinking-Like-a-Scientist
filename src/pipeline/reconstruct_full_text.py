"""Reconstruct full-text inputs for the Claude extraction model-swap.

The original GPT-5.1 extraction pipeline deleted the temporary full-text JSONL
after uploading to the OpenAI Batch API, so the saved 1000batch_api.jsonl has
'[FULL TEXT REMOVED]'. This script recovers full text by:
  1. Parsing titles from the saved batch input
  2. Querying the arXiv API by title to get arXiv IDs
  3. Downloading PDFs and extracting text via PyMuPDF

Usage:
    python src/pipeline/reconstruct_full_text.py [--n-papers 100] [--seed 42]
"""

import argparse
import io
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
import fitz

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GT_BATCH_INPUT = ROOT / "data" / "batches" / "gt_extraction_input" / "1000batch_api.jsonl"
OUTPUT_DIR = ROOT / "data" / "batches" / "batch_suggestions_results" / "claude_extraction"

ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


def load_papers(path: Path) -> list[dict]:
    papers = []
    with open(path) as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def sample_papers(papers: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(papers), size=min(n, len(papers)), replace=False)
    indices.sort()
    return [papers[i] for i in indices]


def parse_title_abstract(paper: dict) -> tuple[str, str]:
    input_text = paper["body"].get("input", "")
    title = ""
    abstract = ""
    title_start = input_text.find("Title: ")
    if title_start >= 0:
        rest = input_text[title_start + 7:]
        title_end = rest.find("\n")
        title = rest[:title_end].strip() if title_end >= 0 else rest.strip()
    abs_start = input_text.find("Abstract: ")
    if abs_start >= 0:
        rest = input_text[abs_start + 10:]
        abs_end = rest.find("\n            Full Text:")
        if abs_end < 0:
            abs_end = rest.find("\n            ")
        abstract = rest[:abs_end].strip() if abs_end >= 0 else rest.strip()
    return title, abstract


def query_arxiv(title: str, sleep: float = 3.0) -> dict | None:
    query = f'ti:"{title}"'
    params = urllib.parse.urlencode({"search_query": query, "max_results": 1})
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8")
    except Exception as e:
        log.warning("arXiv API error for '%s': %s", title[:60], e)
        return None
    finally:
        time.sleep(sleep)

    root = ET.fromstring(xml_data)
    entries = root.findall("atom:entry", NS)
    if not entries:
        return None

    entry = entries[0]
    arxiv_id_raw = entry.findtext("atom:id", default="", namespaces=NS)
    arxiv_id = arxiv_id_raw.split("/abs/")[-1] if "/abs/" in arxiv_id_raw else arxiv_id_raw
    found_title = entry.findtext("atom:title", default="", namespaces=NS).strip()
    found_title = " ".join(found_title.split())
    return {"arxiv_id": arxiv_id, "found_title": found_title}


def download_pdf_text(arxiv_id: str) -> str:
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        resp = requests.get(pdf_url, timeout=30)
        resp.raise_for_status()
        with fitz.open(stream=io.BytesIO(resp.content), filetype="pdf") as doc:
            text = "".join(page.get_text("text") for page in doc)
        return text
    except Exception as e:
        log.warning("PDF extraction failed for %s: %s", arxiv_id, e)
        return ""


def run(n_papers: int, seed: int, sleep: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"full_text_n{n_papers}_seed{seed}.jsonl"

    if output_path.exists():
        existing = {}
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["custom_id"]] = rec
        log.info("Resuming: %d papers already processed in %s", len(existing), output_path)
    else:
        existing = {}

    papers = load_papers(GT_BATCH_INPUT)
    log.info("Loaded %d papers from %s", len(papers), GT_BATCH_INPUT)

    sampled = sample_papers(papers, n_papers, seed)
    log.info("Sampled %d papers (seed=%d)", len(sampled), seed)

    results = list(existing.values())
    done_ids = set(existing.keys())

    matched = sum(1 for r in results if r.get("arxiv_id"))
    has_text = sum(1 for r in results if r.get("full_text"))

    to_process = [p for p in sampled if p["custom_id"] not in done_ids]
    log.info("Already done: %d, remaining: %d", len(done_ids), len(to_process))

    for i, paper in enumerate(to_process):
        custom_id = paper["custom_id"]
        title, abstract = parse_title_abstract(paper)

        log.info("[%d/%d] %s: %s", len(done_ids) + i + 1, len(sampled), custom_id, title[:60])

        result = query_arxiv(title, sleep=sleep)
        if result is None:
            log.warning("  No arXiv match")
            rec = {"custom_id": custom_id, "title": title, "abstract": abstract,
                   "arxiv_id": "", "full_text": ""}
            results.append(rec)
            with open(output_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            continue

        matched += 1
        arxiv_id = result["arxiv_id"]
        log.info("  arXiv ID: %s", arxiv_id)

        full_text = download_pdf_text(arxiv_id)
        if full_text:
            has_text += 1
            log.info("  Extracted %d chars", len(full_text))
        else:
            log.warning("  Empty text")

        rec = {"custom_id": custom_id, "title": title, "abstract": abstract,
               "arxiv_id": arxiv_id, "full_text": full_text}
        results.append(rec)
        with open(output_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    total = len(results)
    log.info("Done. Total: %d, arXiv matched: %d (%.1f%%), has text: %d (%.1f%%)",
             total, matched, 100 * matched / total if total else 0,
             has_text, 100 * has_text / total if total else 0)
    log.info("Output: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-papers", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=3.0)
    args = parser.parse_args()

    log.info("Resolved config:")
    log.info("  n_papers = %d", args.n_papers)
    log.info("  seed = %d", args.seed)
    log.info("  sleep = %.1f", args.sleep)

    run(args.n_papers, args.seed, args.sleep)
