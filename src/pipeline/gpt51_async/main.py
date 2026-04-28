import os
import json
import asyncio
import time
from openai import AsyncOpenAI, OpenAI

from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# Initialize client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from .async_call_llm import async_make_openai_request


def read_data(input_json_file: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(input_json_file, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping line {line_number}: {exc}")
                continue
            if n is not None and len(data) >= n:
                break

    return data

def extract_assistant_json(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # OpenAI Responses
    body = entry.get("response", {}).get("body", {})
    for block in body.get("output", []):
        if block.get("type") == "message":
            for content in block.get("content", []):
                if content.get("type") == "output_text":
                    try:
                        return json.loads(content.get("text", ""))
                    except json.JSONDecodeError:
                        pass

    # Gemini
    candidates = entry.get("response", {}).get("candidates", [])
    for cand in candidates:
        for part in cand.get("content", {}).get("parts", []):
            text = part.get("text", "").strip()
            if text.startswith("```"):
                text = text.removeprefix("```").removesuffix("```").strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

    return None


async def process_entry(entry: Dict[str, Any], client: AsyncOpenAI, semaphore: asyncio.Semaphore):
    async with semaphore:
        assistant_json: Optional[Dict[str, Any]] = None
        response: Optional[Any] = None

        try:
            assistant_json = extract_assistant_json(entry)
            if assistant_json is None:
                return {
                    "id": entry.get("id") or entry.get("response", {}).get("responseId"),
                    "custom_id": entry.get("custom_id") or entry.get("key"),
                    "error": "Could not parse assistant JSON",
                }

            gemini_suggested_datasets = assistant_json.get("gemini_suggested_dataset", [])
            gemini_suggested_models = assistant_json.get("gemini_suggested_model", [])
            gemini_suggested_metrics = assistant_json.get("gemini_suggested_evaluation_metric", [])

            user_prompt = f"""Only use the following fields to make your categorization:
                    Datasets: {gemini_suggested_datasets}
                    Models: {gemini_suggested_models}
                    Metrics: {gemini_suggested_metrics}

                    If multiple datasets, models, or evaluation metrics are listed, classify each one separately and return them as a list of objects.
                    """
            response = await async_make_openai_request(client, user_prompt)

            raw_output = (getattr(response, "output_text", "") or "").strip()
            if not raw_output:
                classification_json: Dict[str, Any] = {"error": "Empty response text"}
            else:
                start = raw_output.find("{")
                end = raw_output.rfind("}") + 1
                if start != -1 and end != -1:
                    try:
                        classification_json = json.loads(raw_output[start:end])
                    except json.JSONDecodeError:
                        classification_json = {"error": "Could not parse JSON"}
                else:
                    classification_json = {"error": "No JSON found in GPT output"}

            return {
                "id": entry.get("id") or entry.get("response", {}).get("responseId"),
                "custom_id": entry.get("custom_id") or entry.get("key"),
                "suggestions": assistant_json,
                "classification": classification_json,
            }
        except Exception as exc:
            return {
                "id": entry.get("id") or entry.get("response", {}).get("responseId"),
                "custom_id": entry.get("custom_id") or entry.get("key"),
                "suggestions": assistant_json,
                "error": str(exc),
            }


async def main(
    input_json_file: str,
    output_json_file: str,
    concurrency_limit: int = 100,
    n: Optional[int] = None,
):
    start_time = time.perf_counter()
    try:
        output_dir = os.path.dirname(output_json_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        data = read_data(input_json_file, n)
        if not data:
            print("No input data found.")
            return []

        semaphore = asyncio.Semaphore(concurrency_limit)

        client = AsyncOpenAI()
        try:
            tasks = [
                asyncio.create_task(
                    process_entry(entry, client, semaphore)
                ) 
                for entry in data
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                await client.close()
            except Exception:
                pass

        cleaned_results: List[Dict[str, Any]] = []
        for entry, result in zip(data, results):
            if isinstance(result, Exception):
                print(f"Entry {entry.get('id')} raised an exception: {result}")
                continue
            if result is not None:
                cleaned_results.append(result)

        with open(output_json_file, "w", encoding="utf-8") as f:
            for record in cleaned_results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Saved classification output: {output_json_file}")
        return cleaned_results
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    asyncio.run(
        main(
            input_json_file="batches/batch_suggestions_results/gemini_suggestions_output.jsonl",
            output_json_file="batches/batch_suggestions_results/gemini_classification_no_web_search.jsonl"
        )
    )
