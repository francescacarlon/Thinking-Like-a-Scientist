"""Deterministic regex-based provider mapping for model names.

Validates the GPT-5.1 pipeline's provider classifications against a
hand-crafted lookup table that requires no LLM judge. Reports agreement
statistics and flags mismatches.

Usage:
    python src/pipeline/deterministic_provider_mapping.py [--output-dir outputs/]
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

PROVIDER_RULES: list[tuple[str, str]] = [
    # Microsoft Research — BERT derivatives (must precede Google BERT rule)
    (r"deberta", "Microsoft Research"),
    (r"codebert", "Microsoft Research"),
    (r"biogpt", "Microsoft Research"),
    (r"phi[-\s]?[0-9]", "Microsoft Research"),
    (r"orca\b", "Microsoft Research"),
    (r"lightgbm", "Microsoft Research"),

    # Hugging Face — BERT derivatives (must precede Google BERT rule)
    (r"distilbert", "Hugging Face"),
    (r"bloom\b|bigscience", "Hugging Face"),
    (r"star\s?coder", "Hugging Face"),
    (r"smol", "Hugging Face"),

    # OpenAI
    (r"gpt[-\s]?[2345]", "OpenAI"),
    (r"chatgpt", "OpenAI"),
    (r"\bo[134][-\s]?(mini|pro)?", "OpenAI"),
    (r"dall[-\s]?e", "OpenAI"),
    (r"whisper", "OpenAI"),
    (r"clip(?!p)", "OpenAI"),
    (r"codex", "OpenAI"),
    (r"text-davinci", "OpenAI"),
    (r"instruct\s?gpt", "OpenAI"),

    # DeepSeek (must precede Qwen for DeepSeek-R1-Distill-Qwen)
    (r"deepseek", "DeepSeek"),

    # Meta AI
    (r"llama[-\s]?[0-9]", "Meta AI"),
    (r"llama\b", "Meta AI"),
    (r"opt[-\s]?[0-9]", "Meta AI"),
    (r"roberta", "Meta AI"),
    (r"segment anything|sam\b", "Meta AI"),
    (r"nllb", "Meta AI"),
    (r"seamless", "Meta AI"),

    # Google DeepMind
    (r"gemini", "Google DeepMind"),
    (r"gemma[-\s]?[0-9]?", "Google DeepMind"),
    (r"^bert$|^bert[-\s]", "Google DeepMind"),
    (r"palm[-\s]?[0-9]?", "Google DeepMind"),
    (r"^t5$|flan[-\s]?t5|ul2|^codet5", "Google DeepMind"),
    (r"albert\b", "Google DeepMind"),
    (r"^vit\b", "Google DeepMind"),
    (r"electra\b", "Google DeepMind"),

    # Anthropic
    (r"claude", "Anthropic"),

    # Alibaba / Qwen team
    (r"qwen", "Alibaba / Qwen team"),
    (r"qwq", "Alibaba / Qwen team"),

    # Mistral AI
    (r"mistral", "Mistral AI"),
    (r"mixtral", "Mistral AI"),
    (r"codestral", "Mistral AI"),
    (r"pixtral", "Mistral AI"),

    # Cohere
    (r"cohere|command[-\s]?r", "Cohere"),

    # Stability AI
    (r"stable\s?diffusion", "Stability AI"),
    (r"stable\s?lm", "Stability AI"),

    # NVIDIA / NeMo
    (r"nemotron", "NVIDIA / NeMo"),
    (r"megatron", "NVIDIA / NeMo"),

    # Databricks / MosaicML
    (r"dbrx", "Databricks / MosaicML"),
    (r"mpt[-\s]?[0-9]", "Databricks / MosaicML"),
]

COMPILED_RULES = [(re.compile(pat, re.IGNORECASE), prov) for pat, prov in PROVIDER_RULES]


def classify_provider(model_name: str) -> str | None:
    for regex, provider in COMPILED_RULES:
        if regex.search(model_name):
            return provider
    return None


def load_classified_models(jsonl_path: Path) -> list[dict[str, str]]:
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            paper_id = rec.get("custom_id", "")
            for m in rec.get("classification", {}).get("model_type", []):
                name = m.get("ModelName", "")
                prov_list = m.get("Provider", [])
                prov = prov_list[0] if isinstance(prov_list, list) and prov_list else str(prov_list)
                if name:
                    records.append({"paper_id": paper_id, "model_name": name, "pipeline_provider": prov})
    return records


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "GT": ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_GT_full_text" / "GT_classification_no_web_search_Fran.jsonl",
        "GPT-5.1": ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_with_pipeline" / "1000_GPT51_batch_responses_classified_converted.jsonl",
        "Gemini": ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_with_pipeline" / "981_gemini_batch_responses_classified_converted.jsonl",
        "DeepSeek": ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_with_pipeline" / "1000_deepseek_batch_responses_classified_converted.jsonl",
    }

    all_rows = []
    for source_name, path in sources.items():
        if not path.exists():
            log.warning("File not found: %s", path)
            continue
        records = load_classified_models(path)
        for r in records:
            r["source"] = source_name
            r["regex_provider"] = classify_provider(r["model_name"])
        all_rows.extend(records)

    df = pd.DataFrame(all_rows)
    matched = df[df["regex_provider"].notna()].copy()
    unmatched = df[df["regex_provider"].isna()].copy()

    matched["agree"] = matched["pipeline_provider"] == matched["regex_provider"]

    log.info("Total model mentions: %d", len(df))
    log.info("Regex-matched: %d (%.1f%%)", len(matched), 100 * len(matched) / len(df))
    log.info("Unmatched (Other/Academic or unknown): %d", len(unmatched))

    summary_rows = []
    for source_name in sources:
        src = matched[matched["source"] == source_name]
        if len(src) == 0:
            continue
        agree_count = int(src["agree"].sum())
        total = len(src)
        pct = 100 * agree_count / total if total else 0
        summary_rows.append({
            "source": source_name,
            "regex_matched": total,
            "agree": agree_count,
            "disagree": total - agree_count,
            "agreement_pct": round(pct, 1),
        })
        log.info("  %s: %d/%d agree (%.1f%%)", source_name, agree_count, total, pct)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "provider_mapping_agreement.csv", index=False)

    mismatches = matched[~matched["agree"]].copy()
    mismatches = mismatches.sort_values(["source", "model_name"])
    mismatches.to_csv(output_dir / "provider_mapping_mismatches.csv", index=False)
    log.info("Mismatches: %d total", len(mismatches))

    if len(mismatches) > 0:
        log.info("Top mismatches:")
        mismatch_counts = mismatches.groupby(["model_name", "pipeline_provider", "regex_provider"]).size().reset_index(name="count")
        mismatch_counts = mismatch_counts.sort_values("count", ascending=False)
        for _, row in mismatch_counts.head(20).iterrows():
            log.info("  %3d× %-30s pipeline=%-25s regex=%-25s",
                     row["count"], row["model_name"], row["pipeline_provider"], row["regex_provider"])

    unmatched_freq = unmatched.groupby("model_name").size().reset_index(name="count").sort_values("count", ascending=False)
    unmatched_freq.to_csv(output_dir / "provider_mapping_unmatched.csv", index=False)
    log.info("Top unmatched models (likely Other/Academic):")
    for _, row in unmatched_freq.head(15).iterrows():
        pipe_prov = unmatched[unmatched["model_name"] == row["model_name"]]["pipeline_provider"].iloc[0]
        log.info("  %3d× %-30s (pipeline: %s)", row["count"], row["model_name"], pipe_prov)

    df.to_csv(output_dir / "provider_mapping_full.csv", index=False)
    log.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    run(args.output_dir)
