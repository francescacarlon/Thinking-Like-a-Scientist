"""Model-swap robustness check: rerun GT entity extraction with Claude Opus 4.6.

Samples 200 papers from the 1,000-paper corpus and submits them to the
Anthropic Message Batches API using the same extraction prompt as the
GPT-5.1 pipeline. This validates that the headline provider-concentration
result is not an artifact of using GPT-5.1 for extraction.

Three subcommands:
  submit   — sample papers and submit batch to Anthropic API
  poll     — check batch status
  retrieve — download results and convert to pipeline-compatible JSONL

Usage:
    # Step 1: submit
    python src/pipeline/batch_claude_extraction.py submit \
        --n-papers 200 --seed 42

    # Step 2: poll (repeat until "ended")
    python src/pipeline/batch_claude_extraction.py poll --batch-id msgbatch_...

    # Step 3: retrieve
    python src/pipeline/batch_claude_extraction.py retrieve --batch-id msgbatch_...

Requires:
    pip install anthropic
    ANTHROPIC_API_KEY in .env or environment
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
GT_BATCH_INPUT = ROOT / "data" / "batches" / "gt_extraction_input" / "1000batch_api.jsonl"
OUTPUT_DIR = ROOT / "data" / "batches" / "batch_suggestions_results" / "claude_extraction"

MODEL = "claude-opus-4-6"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are an academic assistant. Given the title, abstract, and full text of a paper:

1. Generate a single concise research question.
2. Extract:
   - datasets used
   - models used
   - evaluation metrics used

Respond in JSON with:
         {
         "research_question": "...",
         "GroundTruthDatasets": ["..."],
         "GroundTruthModels": ["..."],
         "GroundTruthMetrics": ["..."]
         }

Only report the datasets, models, and metrics used in the experiments and not from the literature review or related work sections.
Each dataset, model and evaluation metric name must be composed from one to three words tops."""


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


def build_batch_requests(papers: list[dict], full_text_map: dict[str, dict] | None = None) -> list[dict]:
    requests = []
    for paper in papers:
        custom_id = paper["custom_id"]
        if full_text_map and custom_id in full_text_map:
            ft = full_text_map[custom_id]
            full_text_section = f"Full Text: {ft['full_text']}" if ft.get("full_text") else ""
            user_prompt = f"""Return JSON with research_question, datasets, models, metrics. Format as valid JSON.

            Title: {ft['title']}
            Abstract: {ft['abstract']}
            {full_text_section}""".strip()
        else:
            user_prompt = paper["body"]["input"]
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": MODEL,
                "max_tokens": MAX_TOKENS,
                "system": SYSTEM_PROMPT,
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            },
        })
    return requests


def cmd_submit(args: argparse.Namespace) -> None:
    try:
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
    except ImportError:
        log.error("anthropic package not installed. Run: pip install anthropic")
        raise SystemExit(1)

    full_text_map = None
    if args.full_text_jsonl:
        ft_path = Path(args.full_text_jsonl)
        if not ft_path.exists():
            log.error("Full-text JSONL not found: %s", ft_path)
            raise SystemExit(1)
        full_text_map = {}
        with open(ft_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("full_text"):
                    full_text_map[rec["custom_id"]] = rec
        log.info("Loaded %d full-text records from %s", len(full_text_map), ft_path)

    papers = load_papers(GT_BATCH_INPUT)
    log.info("Loaded %d papers from %s", len(papers), GT_BATCH_INPUT)

    if full_text_map:
        sampled = [p for p in papers if p["custom_id"] in full_text_map]
        log.info("Using %d papers with full text available", len(sampled))
    else:
        sampled = sample_papers(papers, args.n_papers, args.seed)
        log.info("Sampled %d papers (seed=%d)", len(sampled), args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sampled_path = OUTPUT_DIR / f"sampled_papers_n{len(sampled)}_seed{args.seed}.jsonl"
    with open(sampled_path, "w") as f:
        for p in sampled:
            f.write(json.dumps({"custom_id": p["custom_id"]}) + "\n")
    log.info("Saved sampled paper IDs to %s", sampled_path)

    batch_requests = build_batch_requests(sampled, full_text_map=full_text_map)

    client = anthropic.Anthropic()
    sdk_requests = []
    for req in batch_requests:
        sdk_requests.append(
            Request(
                custom_id=req["custom_id"],
                params=MessageCreateParamsNonStreaming(
                    model=req["params"]["model"],
                    max_tokens=req["params"]["max_tokens"],
                    system=req["params"]["system"],
                    messages=req["params"]["messages"],
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
    log.info("Next: python %s poll --batch-id %s", Path(__file__).name, message_batch.id)


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
    raw_path = OUTPUT_DIR / f"claude_extraction_raw_{batch_id}.jsonl"
    converted_path = OUTPUT_DIR / f"claude_extraction_converted.jsonl"

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

    log.info("Downloaded %d results (%d succeeded, %d errored/expired)", len(raw_results), succeeded, errored)

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
            log.warning("Truncated response for %s (stop_reason=max_tokens)", custom_id)
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
                parsed = {"research_question": "", "GroundTruthDatasets": [], "GroundTruthModels": [], "GroundTruthMetrics": []}

        converted.append({
            "custom_id": custom_id,
            "research_question": parsed.get("research_question", ""),
            "datasets": parsed.get("datasets") or parsed.get("GroundTruthDatasets", []),
            "models": parsed.get("models") or parsed.get("GroundTruthModels", []),
            "metrics": parsed.get("metrics") or parsed.get("GroundTruthMetrics", []),
        })

    with open(converted_path, "w") as f:
        for rec in converted:
            f.write(json.dumps(rec) + "\n")
    log.info("Converted %d records to %s (%d parse errors, %d truncated)", len(converted), converted_path, parse_errors, truncated)

    log.info("Summary:")
    n_datasets = sum(len(r["datasets"]) for r in converted)
    n_models = sum(len(r["models"]) for r in converted)
    n_metrics = sum(len(r["metrics"]) for r in converted)
    log.info("  Papers: %d", len(converted))
    log.info("  Total datasets extracted: %d", n_datasets)
    log.info("  Total models extracted: %d", n_models)
    log.info("  Total metrics extracted: %d", n_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Sample papers and submit batch")
    p_submit.add_argument("--n-papers", type=int, default=200)
    p_submit.add_argument("--seed", type=int, default=42)
    p_submit.add_argument("--full-text-jsonl", type=str, default="",
                          help="Path to full-text JSONL from reconstruct_full_text.py. "
                               "When provided, uses full text instead of redacted input.")

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
