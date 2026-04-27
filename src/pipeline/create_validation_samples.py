"""Create stratified validation samples for annotation of the measurement pipeline.

Generates four sets of CSVs (defaults):
1. validation_extraction.csv — 30 papers for entity extraction P/R
2. validation_classification_{datasets,models,metrics}.csv — 200 entities total for taxonomy validation
3. validation_normalization.csv — 100 entity pairs for fuzzy-merge validation
4. validation_introducedness.csv — 300 entity-paper pairs for introducedness labelling

Usage:
    python src/create_validation_samples.py --output-dir validation/ --seed 42
"""

import argparse
import json
import logging
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from thefuzz import fuzz

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

GT_EXTRACTION = ROOT / "batches/batch_suggestions_results/GT1000batch_6932ad05f04081908c4727f8a46aa838_output.jsonl"
GT_BATCH_INPUT = ROOT / "batches/ready_for_classification/1000batch_api.jsonl"
GT_CLASSIFICATION = ROOT / "batches/ready_for_classification/GT1000batch_responses_classified_Francesca.jsonl"

LLM_SUGGESTION_FILES = {
    "GPT-5.1": ROOT / "batches/batch_suggestions_results/1000SUGG_batch_6932dd7e1c1881908b1a2fa3c8ed7b70_output.jsonl",
    "Gemini": ROOT / "batches/batch_suggestions_results/gemini_suggestions_output.jsonl",
    "DeepSeek": ROOT / "batches/batch_suggestions_results/async_responses_deepseek.jsonl",
}

LLM_CLASSIFICATION_FILES = {
    "GPT-5.1": ROOT / "batches/batch_suggestions_results/classification_with_pipeline/1000_GPT51_batch_responses_classified_converted.jsonl",
    "DeepSeek": ROOT / "batches/batch_suggestions_results/classification_with_pipeline/1000_deepseek_batch_responses_classified_converted.jsonl",
    "Gemini": ROOT / "batches/batch_suggestions_results/classification_with_pipeline/981_gemini_batch_responses_classified_converted.jsonl",
}

DATASET_DIMS = ["Modalities", "TaskTypes", "Domains", "Annotation", "Size", "Granularity", "Linguistic", "CognitiveAffective", "DataQuality"]
MODEL_DIMS = ["Architecture", "TrainingParadigm", "Provider", "Openness", "Size"]
METRIC_DIMS = ["EvType"]


def parse_gt_extraction(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            custom_id = obj["custom_id"]
            body = obj["response"]["body"]
            outputs = body.get("output", [])
            text = None
            for out in outputs:
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        if c.get("type") == "output_text":
                            text = c["text"]
            if text is None:
                continue
            parsed = json.loads(text)
            rq = parsed.get("research_question", "")
            datasets = parsed.get("datasets") or parsed.get("GroundTruthDatasets", [])
            models = parsed.get("models") or parsed.get("GroundTruthModels", [])
            metrics = parsed.get("metrics") or parsed.get("GroundTruthMetrics", [])
            records.append({
                "paper_id": custom_id,
                "research_question": rq,
                "datasets": datasets,
                "models": models,
                "metrics": metrics,
                "n_datasets": len(datasets),
                "n_models": len(models),
                "n_metrics": len(metrics),
                "n_total": len(datasets) + len(models) + len(metrics),
            })
    return records


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


def parse_classification_jsonl(path: Path, source: str) -> list[dict]:
    records = []
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        entries = json.loads(content)
    else:
        entries = [json.loads(line) for line in content.split("\n") if line.strip()]

    for entry in entries:
        custom_id = entry.get("custom_id", "")
        classification = entry.get("classification", {})

        ds_key = "dataset_type" if "dataset_type" in classification else "GroundTruth_dataset_type"
        md_key = "model_type" if "model_type" in classification else "GroundTruth_model_type"
        mt_key = "metric_type" if "metric_type" in classification else "GroundTruth_metric_type"

        for ds in classification.get(ds_key, []):
            name = ds.get("DatasetName", "")
            if not name:
                continue
            rec = {"paper_id": custom_id, "source": source, "entity_type": "dataset", "entity_name": name}
            for dim in DATASET_DIMS:
                rec[f"pipeline_{dim}"] = "; ".join(ds.get(dim, []))
            records.append(rec)

        for md in classification.get(md_key, []):
            name = md.get("ModelName", "")
            if not name:
                continue
            rec = {"paper_id": custom_id, "source": source, "entity_type": "model", "entity_name": name}
            for dim in MODEL_DIMS:
                rec[f"pipeline_{dim}"] = "; ".join(md.get(dim, []))
            records.append(rec)

        for mt in classification.get(mt_key, []):
            name = mt.get("MetricName", "")
            if not name:
                continue
            rec = {"paper_id": custom_id, "source": source, "entity_type": "metric", "entity_name": name}
            for dim in METRIC_DIMS:
                rec[f"pipeline_{dim}"] = "; ".join(mt.get(dim, []))
            records.append(rec)

    return records


def _extract_text_from_response(obj: dict, source: str) -> tuple[str | None, str]:
    custom_id = obj.get("custom_id", "")
    if source == "GPT-5.1":
        body = obj.get("response", {}).get("body", {})
        for out in body.get("output", []):
            if out.get("type") == "message":
                for c in out.get("content", []):
                    if c.get("type") == "output_text":
                        return c["text"], custom_id
    elif source == "Gemini":
        candidates = obj.get("response", {}).get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                raw = parts[0].get("text", "")
                raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
                return raw, custom_id
    elif source == "DeepSeek":
        choices = obj.get("response", {}).get("choices", [])
        if choices:
            raw = choices[0].get("message", {}).get("content", "")
            raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            return raw, custom_id
    return None, custom_id


def _safe_str_list(items: list) -> list[str]:
    return [str(x) for x in items if isinstance(x, (str, int, float))]


def parse_llm_suggestions(path: Path, source: str) -> list[dict]:
    records = []
    prefix_map = {"GPT-5.1": "GPT51", "Gemini": "gemini", "DeepSeek": "deepseek"}
    prefix = prefix_map.get(source, source)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text, custom_id = _extract_text_from_response(obj, source)
            if text is None:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            datasets = _safe_str_list(parsed.get(f"{prefix}_suggested_dataset", []))
            models = _safe_str_list(parsed.get(f"{prefix}_suggested_model", []))
            metrics = _safe_str_list(parsed.get(f"{prefix}_suggested_evaluation_metric", []))
            records.append({
                "paper_id": custom_id,
                "source": source,
                "datasets": datasets,
                "models": models,
                "metrics": metrics,
            })
    return records


def cluster_similar_names_fuzzy(names: list[str], threshold: int = 90) -> list[dict]:
    reps = []
    merge_pairs = []
    for n in sorted(names, key=lambda x: (-len(x), x)):
        name = n.lower().strip()
        if name == "gpt4o3mini":
            continue
        if not reps:
            reps.append(n)
            continue
        best_rep, best_score = None, -1
        for r in reps:
            score = fuzz.token_sort_ratio(n, r)
            if n.startswith(r) or r.startswith(n):
                score = 0
            if score > best_score:
                best_score, best_rep = score, r
        if best_score >= threshold:
            merge_pairs.append({
                "representative": best_rep,
                "variant": n,
                "fuzzy_score": best_score,
                "merged": True,
            })
        else:
            reps.append(n)
    return merge_pairs


def get_near_misses(names: list[str], threshold_low: int = 80, threshold_high: int = 89) -> list[dict]:
    reps = []
    near_misses = []
    for n in sorted(names, key=lambda x: (-len(x), x)):
        name = n.lower().strip()
        if name == "gpt4o3mini":
            continue
        if not reps:
            reps.append(n)
            continue
        best_rep, best_score = None, -1
        for r in reps:
            score = fuzz.token_sort_ratio(n, r)
            if n.startswith(r) or r.startswith(n):
                score = 0
            if score > best_score:
                best_score, best_rep = score, r
        if best_score >= 90:
            pass
        elif threshold_low <= best_score <= threshold_high:
            near_misses.append({
                "representative": best_rep,
                "variant": n,
                "fuzzy_score": best_score,
                "merged": False,
            })
        reps.append(n) if best_score < 90 else None
    return near_misses


def create_extraction_sample(
    gt_records: list[dict],
    titles: dict[str, str],
    n_papers: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = pd.DataFrame(gt_records)
    df["title"] = df["paper_id"].map(titles)
    df["n_total_bin"] = pd.qcut(df["n_total"], q=3, labels=["low", "mid", "high"])

    sampled_indices = []
    for _, g in df.groupby("n_total_bin", observed=True):
        idx = g.sample(n=min(len(g), n_papers // 3 + 1), random_state=int(rng.integers(1e6))).index
        sampled_indices.extend(idx)
    sampled = df.loc[sampled_indices].head(n_papers)

    rows = []
    for _, paper in sampled.iterrows():
        title = paper["title"] or ""
        arxiv_url = f"https://arxiv.org/search/?query={quote_plus(title)}&searchtype=all"
        for etype in ["datasets", "models", "metrics"]:
            entities = paper[etype]
            rows.append({
                "paper_id": paper["paper_id"],
                "title": title,
                "arxiv_search_url": arxiv_url,
                "research_question": paper["research_question"],
                "entity_type": etype.rstrip("s"),
                "extracted_entities": "; ".join(entities) if entities else "",
                "num_extracted": len(entities),
                "annotator_num_correct": "",
                "annotator_num_hallucinated": "",
                "annotator_missed_entities": "",
                "annotator_notes": "",
            })
    return pd.DataFrame(rows)


def create_classification_sample(
    all_classified: list[dict],
    entity_type: str,
    n_entities: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = pd.DataFrame([r for r in all_classified if r["entity_type"] == entity_type])
    if df.empty:
        return pd.DataFrame()

    unique_entities = df.drop_duplicates(subset=["entity_name", "source"])
    entity_counts = unique_entities["entity_name"].value_counts()
    unique_entities = unique_entities.copy()
    unique_entities["freq_rank"] = unique_entities["entity_name"].map(
        lambda x: entity_counts.get(x, 0)
    )
    unique_entities["freq_bin"] = pd.qcut(
        unique_entities["freq_rank"].rank(method="first"),
        q=3,
        labels=["low", "mid", "high"],
    )

    sampled_indices = []
    for _, g in unique_entities.groupby(["source", "freq_bin"], observed=True):
        idx = g.sample(n=min(len(g), max(1, n_entities // 12 + 1)), random_state=int(rng.integers(1e6))).index
        sampled_indices.extend(idx)
    sampled = unique_entities.loc[sampled_indices].head(n_entities)

    dims = {"dataset": DATASET_DIMS, "model": MODEL_DIMS, "metric": METRIC_DIMS}[entity_type]

    rows = []
    for _, row in sampled.iterrows():
        rec = {
            "paper_id": row["paper_id"],
            "source": row["source"],
            "entity_name": row["entity_name"],
        }
        for dim in dims:
            rec[f"pipeline_{dim}"] = row.get(f"pipeline_{dim}", "")
            rec[f"annotator_{dim}"] = ""
        rec["annotator_notes"] = ""
        rows.append(rec)
    return pd.DataFrame(rows)


def create_normalization_sample(
    gt_records: list[dict],
    llm_records: list[dict],
    n_pairs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    all_names: dict[str, set[str]] = {"dataset": set(), "model": set(), "metric": set()}

    for rec in gt_records:
        for d in rec["datasets"]:
            if isinstance(d, str):
                all_names["dataset"].add(d)
        for m in rec["models"]:
            if isinstance(m, str):
                all_names["model"].add(m)
        for m in rec["metrics"]:
            if isinstance(m, str):
                all_names["metric"].add(m)

    for rec in llm_records:
        for d in rec.get("datasets", []):
            if isinstance(d, str):
                all_names["dataset"].add(d)
        for m in rec.get("models", []):
            if isinstance(m, str):
                all_names["model"].add(m)
        for m in rec.get("metrics", []):
            if isinstance(m, str):
                all_names["metric"].add(m)

    n_merged = n_pairs // 2
    n_near = n_pairs - n_merged

    all_merges = []
    all_near_misses = []

    for etype, names in all_names.items():
        merges = cluster_similar_names_fuzzy(list(names))
        for m in merges:
            m["entity_type"] = etype
        all_merges.extend(merges)

        near = get_near_misses(list(names))
        for m in near:
            m["entity_type"] = etype
        all_near_misses.extend(near)

    log.info(f"Found {len(all_merges)} actual merges, {len(all_near_misses)} near-misses (score 80-89)")

    df_merges = pd.DataFrame(all_merges)
    df_near = pd.DataFrame(all_near_misses)

    if len(df_merges) > n_merged:
        df_merges = df_merges.sample(n=n_merged, random_state=int(rng.integers(1e6)))
    if len(df_near) > n_near:
        df_near = df_near.sample(n=n_near, random_state=int(rng.integers(1e6)))

    df_all = pd.concat([df_merges, df_near], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)

    df_all["annotator_correct_decision"] = ""
    df_all["annotator_should_merge"] = ""
    df_all["annotator_notes"] = ""

    return df_all[["entity_type", "representative", "variant", "fuzzy_score", "merged",
                    "annotator_correct_decision", "annotator_should_merge", "annotator_notes"]]


def create_introducedness_sample(
    gt_classified: list[dict],
    titles: dict[str, str],
    n_pairs: int,
    rng: np.random.Generator,
    quotas: dict[str, int] | None = None,
) -> pd.DataFrame:
    quotas = quotas or {"model": 120, "dataset": 100, "metric": 80}

    entity_paper_counts: dict[str, dict[str, int]] = {"dataset": {}, "model": {}, "metric": {}}
    entity_papers: dict[str, dict[str, set]] = {"dataset": {}, "model": {}, "metric": {}}
    entity_provider: dict[str, dict[str, str]] = {"model": {}}

    for rec in gt_classified:
        etype = rec["entity_type"]
        name = rec["entity_name"].lower().strip()
        pid = rec["paper_id"]
        entity_paper_counts[etype][name] = entity_paper_counts[etype].get(name, 0) + 1
        if name not in entity_papers[etype]:
            entity_papers[etype][name] = set()
        entity_papers[etype][name].add(pid)
        if etype == "model":
            prov = rec.get("pipeline_Provider", "")
            if prov and name not in entity_provider["model"]:
                entity_provider["model"][name] = prov

    all_pairs = []
    for rec in gt_classified:
        etype = rec["entity_type"]
        name = rec["entity_name"]
        name_lc = name.lower().strip()
        pid = rec["paper_id"]
        gt_count = entity_paper_counts[etype].get(name_lc, 1)

        is_singleton = gt_count == 1
        title = (titles.get(pid, "") or "").lower()
        title_match = name_lc in title if title else False

        provider = ""
        is_other_academic = False
        if etype == "model":
            provider = entity_provider["model"].get(name_lc, rec.get("pipeline_Provider", ""))
            is_other_academic = "other" in provider.lower() or "academic" in provider.lower()

        all_counts = sorted(entity_paper_counts[etype].values())
        if all_counts:
            decile_edges = np.percentile(all_counts, np.arange(10, 110, 10))
            freq_decile = int(np.searchsorted(decile_edges, gt_count, side="right")) + 1
            freq_decile = min(freq_decile, 10)
        else:
            freq_decile = 1

        all_pairs.append({
            "paper_id": pid,
            "entity_type": etype,
            "entity_name": name,
            "gt_count": gt_count,
            "frequency_decile": freq_decile,
            "is_singleton": is_singleton,
            "title_match": title_match,
            "provider": provider,
            "is_other_academic": is_other_academic,
            "annotator_introducedness": "",
            "annotator_mismatch_type": "",
            "annotator_notes": "",
        })

    df = pd.DataFrame(all_pairs).drop_duplicates(subset=["paper_id", "entity_type", "entity_name"])

    sampled_frames = []
    for etype, quota in quotas.items():
        edf = df[df["entity_type"] == etype].copy()
        if edf.empty:
            continue
        singletons = edf[edf["gt_count"] == 1]
        low_freq = edf[(edf["gt_count"] >= 2) & (edf["gt_count"] <= 5)]
        high_freq = edf[edf["gt_count"] > 5]

        n_sing = max(20, quota * 2 // 5)
        n_low = quota * 2 // 5
        n_high = quota - n_sing - n_low

        parts = []
        for subset, n in [(singletons, n_sing), (low_freq, n_low), (high_freq, n_high)]:
            if len(subset) <= n:
                parts.append(subset)
            else:
                parts.append(subset.sample(n=n, random_state=int(rng.integers(1e6))))
        sampled_frames.append(pd.concat(parts))

    result = pd.concat(sampled_frames, ignore_index=True)
    result = result.sample(frac=1, random_state=int(rng.integers(1e6))).reset_index(drop=True)

    cols = ["paper_id", "entity_type", "entity_name", "gt_count", "frequency_decile",
            "is_singleton", "title_match", "provider", "is_other_academic",
            "annotator_introducedness", "annotator_mismatch_type", "annotator_notes"]
    return result[cols]


def main():
    parser = argparse.ArgumentParser(description="Create validation annotation samples")
    parser.add_argument("--output-dir", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-papers", type=int, default=30)
    parser.add_argument("--n-classification-entities", type=int, default=200)
    parser.add_argument("--n-normalization-pairs", type=int, default=100)
    parser.add_argument("--n-introducedness-pairs", type=int, default=300)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Resolved config:")
    log.info(f"  output_dir = {out_dir}")
    log.info(f"  seed = {args.seed}")
    log.info(f"  n_papers = {args.n_papers}")
    log.info(f"  n_classification_entities = {args.n_classification_entities}")
    log.info(f"  n_normalization_pairs = {args.n_normalization_pairs}")
    log.info(f"  n_introducedness_pairs = {args.n_introducedness_pairs}")

    # --- 1. Extraction validation ---
    log.info("Parsing GT extractions...")
    gt_records = parse_gt_extraction(GT_EXTRACTION)
    log.info(f"  {len(gt_records)} papers parsed")

    log.info("Parsing paper titles...")
    titles = parse_paper_titles(GT_BATCH_INPUT)
    log.info(f"  {len(titles)} titles found")

    log.info("Creating extraction sample...")
    df_extraction = create_extraction_sample(gt_records, titles, args.n_papers, rng)
    df_extraction.to_csv(out_dir / "validation_extraction.csv", index=False)
    log.info(f"  Saved {len(df_extraction)} rows to validation_extraction.csv")

    # --- 2. Classification validation ---
    log.info("Parsing classification data...")
    all_classified = []

    log.info("  Parsing GT classification...")
    gt_class = parse_classification_jsonl(GT_CLASSIFICATION, "GT")
    all_classified.extend(gt_class)
    log.info(f"    {len(gt_class)} classified entities from GT")

    for source, path in LLM_CLASSIFICATION_FILES.items():
        log.info(f"  Parsing {source} classification...")
        recs = parse_classification_jsonl(path, source)
        all_classified.extend(recs)
        log.info(f"    {len(recs)} classified entities from {source}")

    n_per_type = {"dataset": 70, "model": 70, "metric": 60}
    for etype, n in n_per_type.items():
        log.info(f"Creating classification sample for {etype}s (n={n})...")
        df_class = create_classification_sample(all_classified, etype, n, rng)
        fname = f"validation_classification_{etype}s.csv"
        df_class.to_csv(out_dir / fname, index=False)
        log.info(f"  Saved {len(df_class)} rows to {fname}")

    # --- 3. Normalization validation ---
    log.info("Parsing LLM suggestions for normalization...")
    llm_records = []
    for source, path in LLM_SUGGESTION_FILES.items():
        recs = parse_llm_suggestions(path, source)
        llm_records.extend(recs)
        log.info(f"  {len(recs)} papers from {source}")

    log.info("Creating normalization sample...")
    df_norm = create_normalization_sample(gt_records, llm_records, args.n_normalization_pairs, rng)
    df_norm.to_csv(out_dir / "validation_normalization.csv", index=False)
    log.info(f"  Saved {len(df_norm)} rows to validation_normalization.csv")

    # --- 4. Introducedness + mismatch type validation ---
    log.info("Creating introducedness sample...")
    gt_only = [r for r in all_classified if r["source"] == "GT"]
    df_intro = create_introducedness_sample(gt_only, titles, args.n_introducedness_pairs, rng)
    for suffix in ["R1", "R2"]:
        fname = f"validation_introducedness_{suffix}.csv"
        df_intro.to_csv(out_dir / fname, index=False)
        log.info(f"  Saved {len(df_intro)} rows to {fname}")

    log.info("Done. Annotation CSVs written to: %s", out_dir)


if __name__ == "__main__":
    main()
