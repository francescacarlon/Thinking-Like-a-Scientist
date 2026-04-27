"""Compare GPT-5.1 vs Claude Opus 4.6 model-swap results.

Two independent comparisons:
  A. Extraction comparison (94 papers): entity counts, Jaccard, provider distribution
  B. Classification comparison (200 papers): per-dimension agreement and Cohen's kappa

Usage:
    python src/pipeline/compare_model_swap.py [--output-dir outputs/]
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

GT_PATH = ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_GT_full_text" / "GT_classification_no_web_search_Fran.jsonl"
CLAUDE_EXTRACTION = ROOT / "data" / "batches" / "batch_suggestions_results" / "claude_extraction" / "claude_extraction_converted.jsonl"
CLAUDE_CLASSIFICATION = ROOT / "data" / "batches" / "batch_suggestions_results" / "claude_classification" / "claude_classification_converted.jsonl"

PROVIDER_RULES: list[tuple[str, str]] = [
    (r"deberta", "Microsoft Research"),
    (r"codebert", "Microsoft Research"),
    (r"biogpt", "Microsoft Research"),
    (r"phi[-\s]?[0-9]", "Microsoft Research"),
    (r"orca\b", "Microsoft Research"),
    (r"lightgbm", "Microsoft Research"),
    (r"distilbert", "Hugging Face"),
    (r"bloom\b|bigscience", "Hugging Face"),
    (r"star\s?coder", "Hugging Face"),
    (r"smol", "Hugging Face"),
    (r"gpt[-\s]?[2345]", "OpenAI"),
    (r"chatgpt", "OpenAI"),
    (r"\bo[134][-\s]?(mini|pro)?", "OpenAI"),
    (r"dall[-\s]?e", "OpenAI"),
    (r"whisper", "OpenAI"),
    (r"clip(?!p)", "OpenAI"),
    (r"codex", "OpenAI"),
    (r"text-davinci", "OpenAI"),
    (r"instruct\s?gpt", "OpenAI"),
    (r"deepseek", "DeepSeek"),
    (r"llama[-\s]?[0-9]", "Meta AI"),
    (r"llama\b", "Meta AI"),
    (r"opt[-\s]?[0-9]", "Meta AI"),
    (r"roberta", "Meta AI"),
    (r"segment anything|sam\b", "Meta AI"),
    (r"nllb", "Meta AI"),
    (r"seamless", "Meta AI"),
    (r"gemini", "Google DeepMind"),
    (r"gemma[-\s]?[0-9]?", "Google DeepMind"),
    (r"^bert$|^bert[-\s]", "Google DeepMind"),
    (r"palm[-\s]?[0-9]?", "Google DeepMind"),
    (r"^t5$|flan[-\s]?t5|ul2|^codet5", "Google DeepMind"),
    (r"albert\b", "Google DeepMind"),
    (r"^vit\b", "Google DeepMind"),
    (r"electra\b", "Google DeepMind"),
    (r"claude", "Anthropic"),
    (r"qwen", "Alibaba / Qwen team"),
    (r"qwq", "Alibaba / Qwen team"),
    (r"mistral", "Mistral AI"),
    (r"mixtral", "Mistral AI"),
    (r"codestral", "Mistral AI"),
    (r"pixtral", "Mistral AI"),
    (r"cohere|command[-\s]?r", "Cohere"),
    (r"stable\s?diffusion", "Stability AI"),
    (r"stable\s?lm", "Stability AI"),
    (r"nemotron", "NVIDIA / NeMo"),
    (r"megatron", "NVIDIA / NeMo"),
    (r"dbrx", "Databricks / MosaicML"),
    (r"mpt[-\s]?[0-9]", "Databricks / MosaicML"),
]
COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), prov) for pat, prov in PROVIDER_RULES]


def classify_provider(model_name: str) -> str:
    for regex, provider in COMPILED_RULES:
        if regex.search(model_name):
            return provider
    return "Other / Academic"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def get_suggestions(rec: dict) -> dict:
    sug = rec.get("suggestions", {})
    return {
        "datasets": sug.get("datasets") or sug.get("GroundTruthDatasets", []),
        "models": sug.get("models") or sug.get("GroundTruthModels", []),
        "metrics": sug.get("metrics") or sug.get("GroundTruthMetrics", []),
    }


def jaccard(a: list[str], b: list[str]) -> float:
    sa = {s.lower().strip() for s in a if s}
    sb = {s.lower().strip() for s in b if s}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def provider_distribution(models: list[str]) -> Counter:
    counts: Counter = Counter()
    for m in models:
        counts[classify_provider(m)] += 1
    return counts


def normalize_label(val: object) -> str:
    if isinstance(val, list):
        items = sorted(str(v).strip().lower() for v in val if v)
        return "; ".join(items)
    return str(val).strip().lower() if val else ""


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return float("nan")
    all_labels = sorted(set(labels_a) | set(labels_b))
    if len(all_labels) < 2:
        return 1.0 if labels_a == labels_b else 0.0
    n = len(labels_a)
    observed_agree = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected_agree = sum(count_a.get(l, 0) * count_b.get(l, 0) for l in all_labels) / (n * n)
    if expected_agree >= 1.0:
        return 1.0
    return (observed_agree - expected_agree) / (1.0 - expected_agree)


def compare_extraction(gt_by_id: dict[str, dict], claude_ext: list[dict]) -> pd.DataFrame:
    log.info("=== EXTRACTION COMPARISON ===")
    claude_by_id = {r["custom_id"]: r for r in claude_ext}
    common_ids = sorted(set(gt_by_id) & set(claude_by_id))
    log.info("Matched papers: %d", len(common_ids))

    rows = []
    all_gpt_models: list[str] = []
    all_claude_models: list[str] = []

    for cid in common_ids:
        gpt_sug = get_suggestions(gt_by_id[cid])
        claude_rec = claude_by_id[cid]
        claude_sug = {
            "datasets": claude_rec.get("datasets", []),
            "models": claude_rec.get("models", []),
            "metrics": claude_rec.get("metrics", []),
        }
        all_gpt_models.extend(gpt_sug["models"])
        all_claude_models.extend(claude_sug["models"])
        rows.append({
            "paper_id": cid,
            "gpt_datasets": len(gpt_sug["datasets"]),
            "claude_datasets": len(claude_sug["datasets"]),
            "gpt_models": len(gpt_sug["models"]),
            "claude_models": len(claude_sug["models"]),
            "gpt_metrics": len(gpt_sug["metrics"]),
            "claude_metrics": len(claude_sug["metrics"]),
            "jaccard_datasets": jaccard(gpt_sug["datasets"], claude_sug["datasets"]),
            "jaccard_models": jaccard(gpt_sug["models"], claude_sug["models"]),
            "jaccard_metrics": jaccard(gpt_sug["metrics"], claude_sug["metrics"]),
        })

    df = pd.DataFrame(rows)

    log.info("\nEntity counts per paper (mean):")
    for entity in ["datasets", "models", "metrics"]:
        gpt_mean = df[f"gpt_{entity}"].mean()
        claude_mean = df[f"claude_{entity}"].mean()
        log.info("  %s: GPT-5.1=%.1f, Claude=%.1f", entity, gpt_mean, claude_mean)

    log.info("\nJaccard similarity (mean):")
    for entity in ["datasets", "models", "metrics"]:
        j = df[f"jaccard_{entity}"].mean()
        log.info("  %s: %.3f", entity, j)

    gpt_prov = provider_distribution(all_gpt_models)
    claude_prov = provider_distribution(all_claude_models)
    all_providers = sorted(set(gpt_prov) | set(claude_prov))

    log.info("\nProvider distribution (regex-mapped):")
    log.info("  %-25s %8s %8s", "Provider", "GPT-5.1", "Claude")
    for p in all_providers:
        log.info("  %-25s %8d %8d", p, gpt_prov.get(p, 0), claude_prov.get(p, 0))

    gpt_vec = np.array([gpt_prov.get(p, 0) for p in all_providers], dtype=float)
    claude_vec = np.array([claude_prov.get(p, 0) for p in all_providers], dtype=float)
    if len(all_providers) >= 3:
        rho, pval = spearmanr(gpt_vec, claude_vec)
        log.info("\nProvider Spearman rho=%.3f (p=%.4f)", rho, pval)
    else:
        rho, pval = float("nan"), float("nan")

    gpt_total = sum(gpt_prov.values())
    claude_total = sum(claude_prov.values())
    log.info("\nProvider share (%%): regex-matched %d/%d GPT models, %d/%d Claude models",
             gpt_total - gpt_prov.get("Other / Academic", 0), gpt_total,
             claude_total - claude_prov.get("Other / Academic", 0), claude_total)

    return df


def compare_classification(gt_by_id: dict[str, dict], claude_cls: list[dict]) -> pd.DataFrame:
    log.info("\n=== CLASSIFICATION COMPARISON ===")
    claude_by_id = {r["custom_id"]: r for r in claude_cls}
    common_ids = sorted(set(gt_by_id) & set(claude_by_id))
    log.info("Matched papers: %d", len(common_ids))

    model_dims = ["Architecture", "TrainingParadigm", "Provider", "Openness", "Size"]
    dataset_dims = ["Modalities", "TaskTypes", "Domains", "Annotation", "Size",
                    "Granularity", "CognitiveAffective", "DataQuality"]
    metric_dims = ["EvType"]

    dim_labels: dict[str, tuple[list[str], list[str]]] = {}

    for cid in common_ids:
        gpt_cls = gt_by_id[cid].get("classification", {})
        claude_cls_rec = claude_by_id[cid].get("classification", {})

        gpt_models = {m.get("ModelName", "").lower().strip(): m for m in gpt_cls.get("model_type", [])}
        claude_models = {m.get("ModelName", "").lower().strip(): m for m in claude_cls_rec.get("model_type", [])}
        for name in set(gpt_models) & set(claude_models):
            if not name:
                continue
            gm, cm = gpt_models[name], claude_models[name]
            for dim in model_dims:
                key = f"model_{dim}"
                if key not in dim_labels:
                    dim_labels[key] = ([], [])
                dim_labels[key][0].append(normalize_label(gm.get(dim)))
                dim_labels[key][1].append(normalize_label(cm.get(dim)))

        gpt_datasets = {d.get("DatasetName", "").lower().strip(): d for d in gpt_cls.get("dataset_type", [])}
        claude_datasets = {d.get("DatasetName", "").lower().strip(): d for d in claude_cls_rec.get("dataset_type", [])}
        for name in set(gpt_datasets) & set(claude_datasets):
            if not name:
                continue
            gd, cd = gpt_datasets[name], claude_datasets[name]
            for dim in dataset_dims:
                key = f"dataset_{dim}"
                if key not in dim_labels:
                    dim_labels[key] = ([], [])
                dim_labels[key][0].append(normalize_label(gd.get(dim)))
                dim_labels[key][1].append(normalize_label(cd.get(dim)))

        gpt_metrics = {m.get("MetricName", "").lower().strip(): m for m in gpt_cls.get("metric_type", [])}
        claude_metrics = {m.get("MetricName", "").lower().strip(): m for m in claude_cls_rec.get("metric_type", [])}
        for name in set(gpt_metrics) & set(claude_metrics):
            if not name:
                continue
            gmet, cmet = gpt_metrics[name], claude_metrics[name]
            for dim in metric_dims:
                key = f"metric_{dim}"
                if key not in dim_labels:
                    dim_labels[key] = ([], [])
                dim_labels[key][0].append(normalize_label(gmet.get(dim)))
                dim_labels[key][1].append(normalize_label(cmet.get(dim)))

    results = []
    log.info("\nPer-dimension agreement and Cohen's kappa:")
    log.info("  %-30s %6s %8s %6s", "Dimension", "n", "Agree%", "kappa")
    for key in sorted(dim_labels):
        gpt_labels, claude_labels = dim_labels[key]
        n = len(gpt_labels)
        agree = sum(a == b for a, b in zip(gpt_labels, claude_labels))
        agree_pct = 100.0 * agree / n if n > 0 else 0.0
        kappa = cohen_kappa(gpt_labels, claude_labels)
        log.info("  %-30s %6d %7.1f%% %6.3f", key, n, agree_pct, kappa)
        results.append({"dimension": key, "n": n, "agreement_pct": agree_pct, "kappa": kappa})

    entity_counts = {
        "model": sum(1 for k in dim_labels if k.startswith("model_")),
        "dataset": sum(1 for k in dim_labels if k.startswith("dataset_")),
        "metric": sum(1 for k in dim_labels if k.startswith("metric_")),
    }
    if dim_labels:
        sample_key = next(iter(dim_labels))
        n_entities = len(dim_labels[sample_key][0])
    else:
        n_entities = 0

    total_model_matches = len(dim_labels.get("model_Provider", ([], []))[0])
    total_dataset_matches = len(dim_labels.get("dataset_Modalities", ([], []))[0])
    total_metric_matches = len(dim_labels.get("metric_EvType", ([], []))[0])
    log.info("\nEntity name matches: %d models, %d datasets, %d metrics",
             total_model_matches, total_dataset_matches, total_metric_matches)

    return pd.DataFrame(results)


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_records = load_jsonl(GT_PATH)
    gt_by_id = {r["custom_id"]: r for r in gt_records}
    log.info("Loaded %d GT records", len(gt_records))

    claude_ext = load_jsonl(CLAUDE_EXTRACTION)
    log.info("Loaded %d Claude extraction records", len(claude_ext))

    claude_cls = load_jsonl(CLAUDE_CLASSIFICATION)
    log.info("Loaded %d Claude classification records", len(claude_cls))

    ext_df = compare_extraction(gt_by_id, claude_ext)
    ext_df.to_csv(output_dir / "model_swap_extraction_comparison.csv", index=False)
    log.info("Saved extraction comparison to %s", output_dir / "model_swap_extraction_comparison.csv")

    cls_df = compare_classification(gt_by_id, claude_cls)
    cls_df.to_csv(output_dir / "model_swap_classification_comparison.csv", index=False)
    log.info("Saved classification comparison to %s", output_dir / "model_swap_classification_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    args = parser.parse_args()
    run(Path(args.output_dir))
