"""
Analysis and visualization script for batch-extracted dataset/model/metric suggestions.

This script loads either a CSV or JSONL batch output, parses assistant-generated JSON fields (e.g., Gemini suggested
datasets/models/metrics), normalizes and de-duplicates names, and applies fuzzy matching to cluster near-duplicates
while avoiding unsafe prefix merges (e.g., GPT-4 vs GPT-4o). It then computes frequency counts and percentages,
generates distribution bar charts, and exports per-field CSV summaries plus plots into an analysis output folder.
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import ast
import json
import argparse
from glob import glob

from thefuzz import fuzz


#####################################################################
# JSON parsing
#####################################################################

def load_input_file(path):
    """
    Loads either CSV or JSON depending on file extension.
    Returns a dataframe with the required columns.
    """

    if path.lower().endswith(".csv"):
        print(f"Loading CSV: {path}")
        return pd.read_csv(path)
    
    elif path.lower().endswith(".jsonl"):
        print(f"Loading JSONL: {path}")
        return parse_jsonl_batch(path)

    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")


def parse_jsonl_batch(path):
    """
    Parses JSONL file where each line is a separate JSON object.
    Each object should have the same structure as the batch JSON.
    """
    print(f"Parsing JSONL batch file: {path}")
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line {idx}")
                continue

            outputs = item.get("response", {}).get("body", {}).get("output", [])
            assistant_json = None

            # Extract assistant JSON from OpenAI batch output.
            for out in outputs:
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        if "text" in c:
                            raw = c["text"]
                            try:
                                assistant_json = json.loads(raw)
                            except:
                                try:
                                    assistant_json = ast.literal_eval(raw)
                                except:
                                    pass


                if assistant_json:
                    break

            if assistant_json is None:
                print(f"No assistant JSON found for line {idx}")
                continue

            rows.append({
                "datasets": assistant_json.get("datasets",
                            assistant_json.get("GroundTruthDatasets", [])),
                "models": assistant_json.get("models",
                            assistant_json.get("GroundTruthModels", [])),
                "metrics": assistant_json.get("metrics",
                            assistant_json.get("GroundTruthMetrics", []))

            })

    print(f"Extracted {len(rows)} rows from JSONL")
    return pd.DataFrame(rows)


#####################################################################
# Analysis
#####################################################################

def analyze_results_for_model(input_file, model_label):

    output_dir = os.path.join(
        "../analysis_outputs_names",
        "1000_GT_extracted_with_fuzz"
    )
    os.makedirs(output_dir, exist_ok=True)

    df = load_input_file(input_file)
    print(f"\nAnalyzing {model_label}: {len(df)} rows loaded")

    # === Safe list parsing ===
    def try_eval_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    x = ast.literal_eval(value)
                    if isinstance(x, list):
                        return x
                except:
                    pass
        return [value] if value else []

    # === Normalization Helpers ===
    BRACKET_RE = re.compile(r"[\[\(].*?[\]\)]")
    SEP_RE = re.compile(r"[\/;|]")
    PUNCT_RE = re.compile(r"[^\w\s]")
    MULTISPACE_RE = re.compile(r"\s+")

    def normalize_text(s, handle_f1=False, handle_metric_variants=False):
        if s is None:
            return ""
        if not isinstance(s, str):
            s = str(s)

        s = BRACKET_RE.sub(" ", s)
        s = SEP_RE.sub(",", s)
        s = s.replace("_", " ").replace("-", " ")
        s = PUNCT_RE.sub(" ", s)
        s = MULTISPACE_RE.sub(" ", s).strip().lower()

        if handle_f1 and ("f1" in s or "f measure" in s or "f-measure" in s):
            return "f1"

        if handle_metric_variants:
            compact = s.replace(" ", "")

            # Remove leading numeric prefixes like "1accuracy" -> "accuracy", "2rouge" -> "rouge"
            compact = re.sub(r"^\d+(?=[a-z])", "", compact)

            # Aggregate rouge variants (rouge-1/2/L/etc.)
            if compact.startswith("rouge"):
                return "rouge"

            # Aggregate bleu variants (bleu-1/2/3/4/etc.)
            if compact.startswith("bleu"):
                return "bleu"

            # Aggregate Pearson (incl. common misspelling "person")
            if "pearson" in compact or compact == "person" or compact.startswith("person"):
                return "pearson"

            # Aggregate Spearman (incl. rho / spearmanr)
            if "spearman" in compact or compact == "spearmanr" or compact == "spearmanrho":
                return "spearman"

            # Return cleaned metric token (e.g., "1accuracy" -> "accuracy")
            return compact

        s = s.replace(" ", "")

        return s

    def explode_column(series):
        out = []
        for s in series.dropna():
            if isinstance(s, list):
                out.extend(s)
            else:
                parts = re.split(r"[,;/|]+", str(s))
                out.extend([p.strip() for p in parts if p.strip()])
        return pd.Series(out)

    def cluster_similar_names_fuzzy(names, threshold=90):
        reps = []
        mapping = {}
        for n in sorted(names, key=lambda x: (-len(x), x)):
            name = n.lower().strip()

            # SPECIAL EXCEPTION: Split "gpt4o3mini" into gpt4 + o3mini
            if name == "gpt4o3mini":
                mapping[n] = ("gpt4", "o3mini")
                continue

            if not reps:
                reps.append(n)
                mapping[n] = n
                continue

            best_rep, best_score = None, -1
            for r in reps:
                score = fuzz.token_sort_ratio(n, r)

                # Do not merge prefix variants such as gpt4o3 and gpt4o3mini.
                if n.startswith(r) or r.startswith(n):
                    score = 0
                

                if score > best_score:
                    best_score, best_rep = score, r
            if best_score >= threshold:
                mapping[n] = best_rep
            else:
                reps.append(n)
                mapping[n] = n
        return mapping

    def get_counts(series, handle_f1=False, handle_metric_variants=False):
        exploded = explode_column(series)
        exploded = exploded.apply(
            lambda m: normalize_text(
                m,
                handle_f1=handle_f1,
                handle_metric_variants=handle_metric_variants
            )
        )
        return exploded.value_counts()
    
    def to_percentage(series):
        total = series.sum()
        if total == 0:
            return series.copy()
        return (series / total * 100).round(2)

    def plot_distribution(counts, title, filename, top_n=30, ylabel="Count"):
        if counts.empty:
            return
        top = counts.head(top_n)
        plt.figure(figsize=(10, 6))
        sns.barplot(y=top.values, x=top.index, orient="v")
        plt.xlabel("")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=45, ha='right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    ground_truth_cols = {
        "datasets": "GT_Datasets",
        "models": "GT_Models",
        "metrics": "GT_Metrics"
    }

    # Parse all candidate columns
    for col in ground_truth_cols.keys():
        if col in df.columns:
            is_metric = "metric" in col.lower()
            handle_f1 = is_metric
            df[col] = df[col].apply(try_eval_list)
            df[col] = df[col].apply(
                lambda L: [
                    normalize_text(
                        x,
                        handle_f1=handle_f1,
                        handle_metric_variants=is_metric
                    )
                    for x in L if x
                ]
            )

    # Store final results
    results = {}

    # === PROCESS COLUMNS: fuzzy merge BEFORE plotting ===
    for col, pretty in ground_truth_cols.items():
        if col not in df.columns:
            continue

        print(f"Processing {col}: {pretty}")

        counts = get_counts(
            df[col],
            handle_f1=("metric" in col.lower()),
            handle_metric_variants=("metric" in col.lower())
        )

        # Fuzzy cluster here (BEFORE plotting)
        mapping = cluster_similar_names_fuzzy(counts.index.tolist(), threshold=90)

        merged = {}
        for orig, cnt in counts.items():
            rep = mapping.get(orig, orig)
            merged.setdefault(rep, 0)
            merged[rep] += int(cnt)

        merged_series = pd.Series(merged).sort_values(ascending=False)
        percent_series = to_percentage(merged_series)

        # Save in results
        results[pretty] = {
            "counts": merged_series,
            "percent": percent_series
        }


        # Plot FROM MERGED VALUES
        plot_distribution(
            merged_series,
            title=f"{pretty} Distribution (Counts)",
            filename=f"{pretty}_distribution_counts.png",
            ylabel="Count"
        )

        plot_distribution(
            percent_series,
            title=f"{pretty} Distribution (Percent)",
            filename=f"{pretty}_distribution_percent.png",
            ylabel="Percent (%)"
        )

    # === Save all merged counts to CSV ===
    for key, data in results.items():
        safe_name = key.replace(" ", "_")

        counts = data["counts"]
        percent = data["percent"]

        counts_path = os.path.join(output_dir, f"{safe_name}_counts.csv")
        percent_path = os.path.join(output_dir, f"{safe_name}_percentages.csv")
        combined_path = os.path.join(
            output_dir, f"{safe_name}_counts_and_percentages.csv"
        )

        counts.to_csv(counts_path, header=["count"])
        percent.to_csv(percent_path, header=["percent"])

        combined = pd.DataFrame({
            "count": counts,
            "percent": percent
        })

        combined.to_csv(combined_path)

        print(f"Saved: {counts_path}")
        print(f"Saved: {percent_path}")
        print(f"Saved: {combined_path}")




#####################################################################
# MAIN WRAPPER
#####################################################################

def analyze_all_models(input_path):
    analyze_results_for_model(input_path, "GT Extracted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to CSV or JSON file")
    args = parser.parse_args()

    analyze_all_models(args.input)
