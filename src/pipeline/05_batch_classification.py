"""
Batch classification pipeline that converts LLM suggestion outputs into structured taxonomy labels using GPT-5.1.

This script parses a prior JSONL batch (LLM-suggested datasets/models/metrics), builds a new OpenAI Batch API
JSONL where each request asks GPT-5.1 to classify items into a fixed schema (dataset modalities/tasks/domains/
annotation/size/granularity/linguistic/cognitive/data quality; model architecture/training/provider/openness/size;
and metric evaluation type). It then uploads and submits the batch, with utilities for retrying a single request
and for post-processing completed batch outputs into a compact “custom_id + classification JSON” JSONL file.
"""

from openai import OpenAI
import pandas as pd
import json
import time
import os
import random
from dotenv import load_dotenv

load_dotenv()

# Initialize client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

BATCH_INPUT_PATH = "batches/1000_deepseek_classify_without_pipeline.jsonl"

system_prompt = f"""
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

{{
  "dataset_type": [
    {{
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
    }}
  ],

  "model_type": [
    {{
      "ModelName": "...",
      "Architecture": [],
      "TrainingParadigm": [],
      "Provider": [],
      "Openness": [],
      "Size": []
    }}
  ],

  "metric_type": [
    {{
      "MetricName": "...",
      "EvType": []
    }}
  ]
}}


"""
def call_with_retry(client, **kwargs):
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            return client.responses.create(**kwargs)

        except Exception as e:
            err_name = type(e).__name__
            print(f"⚠️ {err_name} (attempt {attempt+1}/{max_retries}): {e}")

            sleep_time = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            print(f"⏳ Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    raise Exception("❌ Failed after maximum retries.")


def build_batch_input_jsonl(upstream_batch_output_jsonl, batch_input_path, n=None):
    data = []
    with open(upstream_batch_output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if n is not None:
        data = data[:n]

    os.makedirs(os.path.dirname(batch_input_path), exist_ok=True)

    written = 0

    with open(batch_input_path, "w", encoding="utf-8") as f:
        for i, entry in enumerate(data, start=1):
            assistant_json = None

            parsed = entry.get("parsed") or entry.get("response", {}).get("parsed")
            if isinstance(parsed, dict):
                assistant_json = parsed
            else:
                # DeepSeek text responses are stored here when parsed JSON is absent.
                text = (
                    entry.get("response", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                )

                if text:
                    text = text.strip()

                    # Remove ```json ... ``` fences if present.
                    if text.startswith("```"):
                        first_nl = text.find("\n")
                        if first_nl != -1:
                            text = text[first_nl + 1 :]
                        if text.endswith("```"):
                            text = text[:-3].strip()

                    # Extract JSON object even if there is extra surrounding text.
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start != -1 and end > start:
                        try:
                            assistant_json = json.loads(text[start:end])
                        except json.JSONDecodeError:
                            assistant_json = None

            if assistant_json is None:
                print("⚠️ Skipping: Could not parse assistant JSON")
                continue

            deepseek_suggested_datasets = assistant_json.get("deepseek_suggested_dataset", [])
            deepseek_suggested_models = assistant_json.get("deepseek_suggested_model", [])
            deepseek_suggested_metrics = assistant_json.get("deepseek_suggested_evaluation_metric", [])

            user_prompt = f"""Only use the following fields to make your categorization:
                    Suggested Datasets: {deepseek_suggested_datasets}
                    Suggested Models: {deepseek_suggested_models}
                    Suggested Metrics: {deepseek_suggested_metrics}

                    
                    If multiple datasets, models, or evaluation metrics are listed, classify each one separately and return them as a list of objects.
                    """

            # batch request
            batch_request = {
                "custom_id": entry.get("custom_id") or entry.get("id") or f"row-{i:06d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-5.1-2025-11-13",
                    "instructions": system_prompt,
                    "input": user_prompt,
                    "reasoning": {"effort": "medium"}
                }
            }

            f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
            written += 1


    print(f"✅ Wrote {written} requests → {batch_input_path}")
    if written == 0:
        raise RuntimeError(
            "No batch requests written (all rows skipped). Fix parsing before submitting."
        )


# this worked with GPT-5.1 batch
# def submit_batch(batch_input_path: str) -> str:
#     # Upload input file
#     batch_file = openai_client.files.create(
#         file=open(batch_input_path, "rb"),
#         purpose="batch",
#     )
# 
#     # Create batch job
#     batch = openai_client.batches.create(
#         input_file_id=batch_file.id,
#         endpoint="/v1/responses",
#         completion_window="24h",
#         metadata={"description": "GPT-5.1 classify with pipeline"},
#     )
# 
#     print("✅ Batch created:", batch.id)
#     return batch.id

# try this for Gemini batch
def submit_batch(batch_input_path: str) -> str:
    # Upload input file (force .jsonl filename + mimetype)
    with open(batch_input_path, "rb") as fh:
        batch_file = openai_client.files.create(
            file=(os.path.basename(batch_input_path), fh, "application/jsonl"),
            purpose="batch",
        )

    # Create batch job
    batch = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": "deepseek suggestions classify without pipeline"},
    )

    print("✅ Batch created:", batch.id)
    return batch.id



def parse_batch_output_to_classified_jsonl(batch_output_jsonl: str, final_output_jsonl: str):
    os.makedirs(os.path.dirname(final_output_jsonl), exist_ok=True)

    with open(batch_output_jsonl, "r", encoding="utf-8") as f_in, open(final_output_jsonl, "w", encoding="utf-8") as f_out:
        kept = 0
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # Batch output keeps your custom_id and includes response.body
            custom_id = entry.get("custom_id")

            body = entry.get("response", {}).get("body", {})
            msg_outputs = body.get("output", [])

            raw_text = None
            for block in msg_outputs:
                if block.get("type") == "message":
                    for content in block.get("content", []):
                        if content.get("type") == "output_text":
                            raw_text = content.get("text") #  for GPT51 suggestions
                            break

            classification_json = {"error": "No output_text found"}
            if raw_text:
                raw_text = raw_text.strip()
                # If model returns extra text, extract the JSON object
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end != -1:
                    try:
                        classification_json = json.loads(raw_text[start:end])
                    except json.JSONDecodeError:
                        classification_json = {"error": "Could not parse JSON"}

            record = {
                "custom_id": custom_id,
                "classification": classification_json,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ Saved {kept} classified records → {final_output_jsonl}")


# trial
def run_first_request_from_jsonl(batch_input_path: str):
    with open(batch_input_path, "r", encoding="utf-8") as f:
        first = json.loads(next(f))

    body = first["body"]
    resp = call_with_retry(
        openai_client,
        model=body["model"],
        instructions=body["instructions"],
        input=body["input"],
        reasoning=body.get("reasoning", {"effort": "medium"}),
    )

    # Print model output text
    out = None
    for block in resp.output:
        if block.type == "message":
            for c in block.content:
                if c.type == "output_text":
                    out = c.text
                    break
    print(out)



# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # 1) Build the batch input JSONL (replaces your old per-entry responses.create loop)
    build_batch_input_jsonl(
        upstream_batch_output_jsonl="batches/batch_suggestions_results/async_responses_deepseek.jsonl",
        batch_input_path=BATCH_INPUT_PATH,
        n=None,

    )

    # 2) Submit the batch (creates the server-side batch job)
    # batch_id = run_first_request_from_jsonl(BATCH_INPUT_PATH)
    batch_id = submit_batch(BATCH_INPUT_PATH)
    print("Batch submitted. ID:", batch_id)
