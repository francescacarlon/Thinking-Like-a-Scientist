"""Model-swap robustness check: rerun taxonomy classification with Claude Opus 4.6.

Takes the same GPT-5.1-extracted entities and classifies them with Claude
using the identical taxonomy prompt. This isolates the classifier variable:
same entities, different model.

Three subcommands:
  submit   — build classification requests and submit batch
  poll     — check batch status
  retrieve — download results and convert to pipeline-compatible JSONL

Usage:
    python src/pipeline/batch_claude_classification.py submit [--n-papers 200] [--seed 42]
    python src/pipeline/batch_claude_classification.py poll [--batch-id msgbatch_...]
    python src/pipeline/batch_claude_classification.py retrieve [--batch-id msgbatch_...]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
GT_CLASSIFIED = ROOT / "data" / "batches" / "batch_suggestions_results" / "classification_GT_full_text" / "GT_classification_no_web_search_Fran.jsonl"
OUTPUT_DIR = ROOT / "data" / "batches" / "batch_suggestions_results" / "claude_classification"

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """
1. dataset_type. For each dataset mentioned in `dataset`, identify one or more subcategories under each of the following dimensions:

## Modalities
- Text, Audio, Image, Video, Time series, Graph, Spatial, Multimodal

## Task Types
- Classification, Regression, Sequence labeling, Generation, Summarization, Translation, Question answering, Reasoning, Dialogue, Object detection, Forecasting, Retrieval, Alignment, Multimodal integration, Clustering, Reinforcement learning

## Domains
- General, Media, Scientific / academic, Healthcare, Legal, Economics, Social, Geospatial, Robotics, Vision, Entertainment, Education, Infrastructure, Ontology, Biology, Chemistry, Environmental

## Annotation
- Fully Supervised, Weakly Supervised, Self-Supervised, Semi-Supervised, Reinforcement Feedback, Crowdsourced, Expert Annotations

## Size
- Small: < 10,000 items, Medium: 10,000 – 100,000 items, Large: > 100,000 items

You must choose exactly one of the following categories. Do not include any explanations or item counts.
Allowed values:
- "Small"
- "Medium"
- "Large"

## Granularity
- Document-level, Sentence-level, Token-level, Frame-level, Pixel-level, Object-level


## Linguistic
Classify each dataset under one of these linguistic types:

- Monolingual
- Multilingual
- Cross-lingual

If the dataset is **Monolingual**, specify the language (choose one from the list below).
If **Multilingual**, specify the languages included.
If **Cross-lingual**, specify the alignment or language pair (e.g., "English ↔ French").

Return the value in this exact format:
- Linguistic: ["Monolingual: English"]
or
- Linguistic: ["Multilingual: English, French, German"]
or
- Linguistic: ["Cross-lingual: English ↔ French"]

Do **not** use nested objects or additional fields.
Always follow this pattern: `"Linguistic": ["<Type>: <Language(s)>"]`

### Available languages
English, Chinese, Spanish, French, German, Russian, Portuguese,
Italian, Dutch, Arabic, Japanese, Korean, Turkish, Polish,
Vietnamese, Indonesian, Hebrew, Swedish, Czech, Hungarian, Other


## Cognitive / Affective Dimensions
- Attention, Memory, Problem Solving, Reasoning, Decision Making, Perception, Learning, Cognitive Load, Emotion, Empathy, Theory of Mind, Social Reasoning, Moral Cognition, Personality

## Data Quality
- Noisy, Curated

---
2. model_type. For each model mentioned in `model`, classify it using ONLY the following categories:

## Model Architectures
- Transformer (Encoder / Decoder / Encoder–Decoder)
- Generative
- Convolutional (CNN)
- Recurrent (RNN / LSTM)
- Graph Neural Network (GNN)
- Tree-based
- Linear
- Kernel Models
- Probabilistic
- Reinforcement Learning (RL)

## Training Paradigms
- Supervised Learning
- Self-supervised Learning
- Unsupervised Learning
- Reinforcement Learning
- Multi-task Learning
- Few-shot Learning
- Zero-shot Learning
- Fine-tuning
- Retrieval-Augmented Generation (RAG)

## Provider
- OpenAI, Anthropic, Meta AI, Google DeepMind, Mistral AI, Alibaba / Qwen team,
  Cohere, Hugging Face, Stability AI, Microsoft Research, NVIDIA / NeMo,
  Databricks / MosaicML, DeepSeek, Other / Academic

## Openness
- Closed
- Open

## Size
- Small (<1B), Medium (1–10B), Large (10–100B), Extra-Large (>100B)

---
3. evaluation_metric. For each evaluation metric mentioned in `evaluation_metric`, classify it **only under one of the following categories**:

## Evaluation Type
- Accuracy
- Ranking
- Regression
- Continuous Prediction
- Probability
- Uncertainty
- Fairness
- Safety
- Efficiency / Latency
- Explainability
- Robustness
- User Experience


---

Return your output strictly in this JSON format:

{
  "dataset_type": [
    {
      "DatasetName": "...",
      "Modalities": [],
      "TaskTypes": [],
      "Domains": [],
      "Annotation": [],
      "Size": [],
      "Granularity": [],
      "Linguistic": [],
      "CognitiveAffective": [],
      "DataQuality": []
    }
  ],

  "model_type": [
    {
      "ModelName": "...",
      "Architecture": [],
      "TrainingParadigm": [],
      "Provider": [],
      "Openness": [],
      "Size": []
    }
  ],

  "metric_type": [
    {
      "MetricName": "...",
      "EvType": []
    }
  ]
}
"""


def load_gt_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def sample_records(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(records), size=min(n, len(records)), replace=False)
    indices.sort()
    return [records[i] for i in indices]


def build_user_prompt(record: dict) -> str:
    sug = record.get("suggestions", {})
    datasets = sug.get("datasets") or sug.get("GroundTruthDatasets", [])
    models = sug.get("models") or sug.get("GroundTruthModels", [])
    metrics = sug.get("metrics") or sug.get("GroundTruthMetrics", [])
    return f"""Only use the following fields to make your categorization:
                    Suggested Datasets: {datasets}
                    Suggested Models: {models}
                    Suggested Metrics: {metrics}


                    If multiple datasets, models, or evaluation metrics are listed, classify each one separately and return them as a list of objects.
                    """


def cmd_submit(args: argparse.Namespace) -> None:
    try:
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        raise SystemExit(1)

    records = load_gt_records(GT_CLASSIFIED)
    log.info("Loaded %d records from %s", len(records), GT_CLASSIFIED)

    sampled = sample_records(records, args.n_papers, args.seed)
    log.info("Sampled %d records (seed=%d)", len(sampled), args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled_path = OUTPUT_DIR / f"sampled_papers_n{len(sampled)}_seed{args.seed}.jsonl"
    with open(sampled_path, "w") as f:
        for r in sampled:
            f.write(json.dumps({"custom_id": r["custom_id"]}) + "\n")
    log.info("Saved sampled paper IDs to %s", sampled_path)

    client = anthropic.Anthropic()
    sdk_requests = []
    for record in sampled:
        custom_id = record["custom_id"]
        user_prompt = build_user_prompt(record)
        sdk_requests.append(
            Request(
                custom_id=custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                ),
            )
        )

    log.info("Submitting batch of %d requests to Anthropic API...", len(sdk_requests))
    message_batch = client.messages.batches.create(requests=sdk_requests)
    log.info("Batch created: %s", message_batch.id)
    log.info("Status: %s", message_batch.processing_status)

    batch_id_path = OUTPUT_DIR / "latest_batch_id.txt"
    with open(batch_id_path, "w") as f:
        f.write(message_batch.id)
    log.info("Batch ID saved to %s", batch_id_path)


def cmd_poll(args: argparse.Namespace) -> None:
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        raise SystemExit(1)

    batch_id = args.batch_id
    if not batch_id:
        id_file = OUTPUT_DIR / "latest_batch_id.txt"
        if id_file.exists():
            batch_id = id_file.read_text().strip()
        else:
            log.error("No --batch-id provided and no latest_batch_id.txt found")
            raise SystemExit(1)

    client = anthropic.Anthropic()

    if args.wait:
        while True:
            batch = client.messages.batches.retrieve(batch_id)
            log.info("Status: %s | succeeded=%d errored=%d expired=%d processing=%d",
                     batch.processing_status,
                     batch.request_counts.succeeded,
                     batch.request_counts.errored,
                     batch.request_counts.expired,
                     batch.request_counts.processing)
            if batch.processing_status == "ended":
                log.info("Batch complete!")
                break
            time.sleep(60)
    else:
        batch = client.messages.batches.retrieve(batch_id)
        log.info("Batch %s", batch_id)
        log.info("Status: %s", batch.processing_status)
        log.info("Counts: succeeded=%d errored=%d expired=%d processing=%d",
                 batch.request_counts.succeeded,
                 batch.request_counts.errored,
                 batch.request_counts.expired,
                 batch.request_counts.processing)
        if batch.processing_status != "ended":
            log.info("Still processing. Rerun with --wait to block until done.")


def cmd_retrieve(args: argparse.Namespace) -> None:
    try:
        import anthropic
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        raise SystemExit(1)

    batch_id = args.batch_id
    if not batch_id:
        id_file = OUTPUT_DIR / "latest_batch_id.txt"
        if id_file.exists():
            batch_id = id_file.read_text().strip()
        else:
            log.error("No --batch-id provided and no latest_batch_id.txt found")
            raise SystemExit(1)

    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        log.error("Batch %s not yet ended (status: %s)", batch_id, batch.processing_status)
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"claude_classification_raw_{batch_id}.jsonl"
    converted_path = OUTPUT_DIR / "claude_classification_converted.jsonl"

    log.info("Downloading results for batch %s...", batch_id)
    succeeded = 0
    errored = 0
    raw_results = []

    for result in client.messages.batches.results(batch_id):
        raw_results.append(result)
        if result.result.type == "succeeded":
            succeeded += 1
        else:
            errored += 1

    log.info("Downloaded %d results (%d succeeded, %d errored/expired)",
             len(raw_results), succeeded, errored)

    with open(raw_path, "w") as f:
        for r in raw_results:
            f.write(json.dumps(json.loads(r.model_dump_json())) + "\n")
    log.info("Raw results saved to %s", raw_path)

    converted = []
    parse_errors = 0
    truncated = 0
    for r in raw_results:
        if r.result.type != "succeeded":
            continue
        custom_id = r.custom_id
        if r.result.message.stop_reason == "max_tokens":
            log.warning("Truncated response for %s", custom_id)
            truncated += 1
        text = ""
        for block in r.result.message.content:
            if block.type == "text":
                text += block.text
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start:end])
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None
            if parsed is None:
                log.warning("JSON parse error for %s", custom_id)
                parse_errors += 1
                parsed = {"dataset_type": [], "model_type": [], "metric_type": []}

        converted.append({
            "custom_id": custom_id,
            "classification": parsed,
        })

    with open(converted_path, "w") as f:
        for rec in converted:
            f.write(json.dumps(rec) + "\n")
    log.info("Converted %d records to %s (%d parse errors, %d truncated)",
             len(converted), converted_path, parse_errors, truncated)

    n_datasets = sum(len(r["classification"].get("dataset_type", [])) for r in converted)
    n_models = sum(len(r["classification"].get("model_type", [])) for r in converted)
    n_metrics = sum(len(r["classification"].get("metric_type", [])) for r in converted)
    log.info("Summary: %d papers, %d dataset labels, %d model labels, %d metric labels",
             len(converted), n_datasets, n_models, n_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Build classification requests and submit batch")
    p_submit.add_argument("--n-papers", type=int, default=200)
    p_submit.add_argument("--seed", type=int, default=42)

    p_poll = sub.add_parser("poll", help="Check batch status")
    p_poll.add_argument("--batch-id", type=str, default="")
    p_poll.add_argument("--wait", action="store_true", help="Block until batch completes")

    p_retrieve = sub.add_parser("retrieve", help="Download and convert results")
    p_retrieve.add_argument("--batch-id", type=str, default="")

    args = parser.parse_args()

    if args.command == "submit":
        cmd_submit(args)
    elif args.command == "poll":
        cmd_poll(args)
    elif args.command == "retrieve":
        cmd_retrieve(args)


if __name__ == "__main__":
    main()
