"""
Second-stage Batch API pipeline that takes research questions from a prior batch output and enriches them with
actionable experiment suggestions.

The script parses each completed batch result to extract the `research_question`, then generates a new JSONL batch
input for LLMs to propose suitable datasets, models/LLMs, evaluation metrics, and a short bullet-point pipeline.
It submits the new batch, polls until completion, and downloads the resulting JSONL into a dedicated output folder.
"""

import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# CONFIG
# ============================================================
PREVIOUS_BATCH_OUTPUT = "batches/1000batch_6932ad05f04081908c4727f8a46aa838_output.jsonl"  # OUTPUT FILE from previous batch
NEXT_BATCH_INPUT = "batches/batch_calls_for_suggestions.jsonl"
OUTPUT_DIR = "batches/batch_suggestions_results"
MODEL_NAME = "gpt-5.1-2025-11-13"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# 1. Load research questions from precise structure of your batch output
# ============================================================
def load_research_questions(jsonl_file):
    questions = []
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            custom_id = row.get("custom_id")

            try:
                # Extract GPT output text containing the JSON
                message_block = row["response"]["body"]["output"][1]
                text = message_block["content"][0]["text"]

                parsed = json.loads(text)
                rq = parsed["research_question"]

                questions.append((custom_id, rq))

            except Exception as e:
                print(f"⚠️ Could not parse row {custom_id}: {e}")

    return questions


# ============================================================
# 2. Build JSONL batch request file for GPT-5.1 suggestions
# ============================================================
base_prompt = """
You are an expert AI research assistant.
Given the following research question:

"{research_question}"

Please suggest one or more:
1. suitable datasets to address the question.
2. appropriate machine learning models, architectures, or Large Language Models to use.
3. relevant evaluation metrics for measuring the model's performance.
4. A straightforward pipeline or methodology for solving it.

Only respond with one or more specific dataset names, one or more specific model names, one or more specific evaluation metric names and keep the pipeline structured and short.
Each dataset, model and evaluation metric name must be composed from one to three words tops.
In the pipeline, explain how you want to run the experiment to solve the research question in bullet points.

Respond in valid JSON with keys:
{{
"GPT51_suggested_dataset": ["..."],
"GPT51_suggested_model": ["..."],
"GPT51_suggested_evaluation_metric": ["..."],
"GPT51_suggested_pipeline": "..."
}}
"""

user_prompt = f""" Return JSON categorization objects for each of the following fields:
GPT51_suggested_dataset, GPT51_suggested_model, GPT51_suggested_evaluation_metric, GPT51_suggested_pipeline
"""

def build_batch_input(questions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (custom_id, rq) in enumerate(questions):
            system_prompt = base_prompt.format(research_question=rq)

            row = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL_NAME,
                    "reasoning": {"effort": "medium"},
                    "input": user_prompt,
                    "instructions": system_prompt
                }
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"📄 New batch file created → {output_file}")


# ============================================================
# 3. Submit batch
# ============================================================
def submit_batch(batch_file):
    # 1. Upload JSONL file
    uploaded = client.files.create(
        file=open(batch_file, "rb"),
        purpose="batch"
    )

    # 2. Create batch using file ID
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h"
    )

    print(f"🚀 Batch submitted → batch_id = {batch.id}")
    return batch.id


# ============================================================
# 4. Poll until complete
# ============================================================
def wait_for_batch(batch_id):
    while True:
        batch = client.batches.retrieve(batch_id)
        print(f"⏳ Status: {batch.status}")
        if batch.status in ["completed", "failed", "expired"]:
            return batch
        time.sleep(10)


# ============================================================
# 5. Download output file
# ============================================================
def download_batch_output(batch):
    if not batch.output_file_id:
        print("❌ Batch failed or no output file.")
        return

    output_path = os.path.join(OUTPUT_DIR, f"{batch.id}_output.jsonl")
    content = client.files.content(batch.output_file_id)

    with open(output_path, "wb") as f:
        f.write(content)

    print(f"📥 Downloaded batch results → {output_path}")
    return output_path


# ============================================================
# MAIN PIPELINE
# ============================================================
if __name__ == "__main__":
    print("🔍 Extracting research questions...")
    questions = load_research_questions(PREVIOUS_BATCH_OUTPUT)
    print(f"   → {len(questions)} research questions loaded")

    print("\n📝 Building new batch request file...")
    build_batch_input(questions, NEXT_BATCH_INPUT)

    print("\n🚀 Submitting new batch...")
    batch_id = submit_batch(NEXT_BATCH_INPUT)

    print("\n⏳ Waiting for batch to finish...")
    batch = wait_for_batch(batch_id)

    print("\n📥 Downloading...")
    download_batch_output(batch)

    print("\n🎯 DONE — your next dataset/model/metric batch is complete.")
