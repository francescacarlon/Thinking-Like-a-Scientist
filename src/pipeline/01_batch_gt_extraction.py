"""
Batch pipeline for building an arXiv LLM-paper dataset with OpenAI Batch API enrichment.

This script (1) scrapes recent arXiv CS papers with LLM-related titles, (2) downloads and extracts PDF text,
(3) submits papers to the OpenAI Batch API to generate a concise research question and extract “ground truth”
datasets/models/metrics from the experimental sections, and (4) merges results and saves a final CSV.
It supports resuming from an existing CSV and uses a redacted on-disk JSONL while uploading full text via
a temporary file.
"""

import os
import json
import time
import random
import pandas as pd
import requests
import io
import fitz
import arxiv
import csv
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv


# ============================================================
# Batch-Based Framework for arXiv → Batch API → CSV Pipelines
# ============================================================

class BatchFramework:
    def __init__(self, api_key, category, years, papers_per_year,
                 model_name, batch_folder, base_file=None):

        load_dotenv()
        self.client = OpenAI(api_key=api_key)

        self.category = category
        self.years = years
        self.papers_per_year = papers_per_year
        self.model_name = model_name
        self.batch_folder = batch_folder
        os.makedirs(batch_folder, exist_ok=True)
        random.seed(42)

        # CSV file to store final results
        if base_file is None:
            self.base_file = f"data/arxiv_{category}_{years}_{model_name}_solution_{papers_per_year}_papers.csv"
        else:
            self.base_file = base_file

        self.df_existing, self.existing_titles = self.load_existing_data()



    # ============================================================
    # Load existing CSV
    # ============================================================

    def load_existing_data(self):
        if os.path.exists(self.base_file):
            df = pd.read_csv(self.base_file)
            df.columns = df.columns.str.strip()
            if 'Title' not in df.columns:
                df = pd.DataFrame(columns=["Title", "Abstract", "MainCategory", "SubCategory", "Year"])
            titles = set(df['Title'].astype(str).str.strip())
            print(f"📁 Loaded {len(df)} existing papers")
        else:
            df = pd.DataFrame(columns=["Title", "Abstract", "MainCategory", "SubCategory", "Year"])
            titles = set()
            print("No existing CSV found — starting fresh.")
        return df, titles



    # ============================================================
    # PDF extraction 
    # ============================================================

    def extract_full_text(self, pdf_url):
        if not pdf_url or not str(pdf_url).startswith('http'):
            return ''
        try:
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            with fitz.open(stream=io.BytesIO(response.content), filetype='pdf') as doc:
                text = ''.join(page.get_text('text') for page in doc)
            return text
        except Exception as e:
            print(f"⚠️ PDF extraction error: {e}")
            return ''



    # ============================================================
    # Fetch arXiv Papers - NO MODEL CALLS HERE 
    # ============================================================

    def fetch_papers_from_arxiv(self, year):
        """Scrapes arXiv for papers published in the last 6 months,
        but only keeps papers whose published.year == year."""

        data = []

        # Check how many papers you already have for this year
        existing_year = self.df_existing[self.df_existing['Year'] == year]
        if len(existing_year) >= self.papers_per_year:
            print(f"✅ Already have {len(existing_year)} papers for {year}")
            return data

        needed = self.papers_per_year - len(existing_year)

        # --- Compute 6 month window ---
        today = datetime.utcnow()
        six_months_ago = today - timedelta(days=182)
        start = six_months_ago.strftime("%Y%m%d%H%M")
        end   = today.strftime("%Y%m%d%H%M")

        # --- Query arXiv within that window ---
        query = (
            f'(ti:"LLM*" OR ti:"Large Language Model*") '
            f'AND cat:cs.* '
            f'AND submittedDate:[{start} TO {end}]'
        )

        client_arxiv = arxiv.Client()
        search = arxiv.Search(query=query, max_results=1001,
                            sort_by=arxiv.SortCriterion.SubmittedDate)
        papers = list(client_arxiv.results(search))

        if not papers:
            print("⚠️ No papers found for this query.")
            return data

        # Filter only papers from the specified year
        papers_for_year = [
            p for p in papers
            if p.published.year == year
        ]

        if not papers_for_year:
            print(f"⚠️ No papers published in year {year} found in the last 6 months window.")
            return data

        # Select up to the number needed
        selected = random.sample(papers_for_year, min(needed, len(papers_for_year)))

        # --- Extract data for each selected paper ---
        for result in selected:
            title = result.title.strip().replace('\n', ' ')

            if title in self.existing_titles:
                continue

            abstract = result.summary.replace('\n', ' ').strip()
            pdf_url = result.pdf_url
            full_text = self.extract_full_text(pdf_url)

            data.append({
                'Title': title,
                'Abstract': abstract,
                'FullText': full_text,
                'MainCategory': (result.primary_category or '').split('.')[0],
                'SubCategory': (result.primary_category or '').split('.')[1] if '.' in (result.primary_category or '') else '',
                'Year': result.published.year,
                'PDFURL': pdf_url
            })

            time.sleep(1)

        return data



    # ============================================================
    # Convert Ground Truth arrays to JSON lists for CSV
    # ============================================================

    def convert_ground_truths_to_json(self, df):
        for col in ['GroundTruthDatasets','GroundTruthModels','GroundTruthMetrics']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda val: json.dumps([v.strip() for v in str(val).split(',') if v.strip()])
                    if pd.notna(val) and str(val).strip() not in ['','Unknown']
                    else json.dumps([])
                )
        return df



    # ============================================================
    # Save final CSV
    # ============================================================

    def save_csv(self, df, filename=None):
        # Ensure JSON lists for ground truth columns
        df = self.convert_ground_truths_to_json(df)

        if filename is None:
            base, ext = os.path.splitext(self.base_file)
            filename = f"{base}_with_ground_truth.csv"

        df.to_csv(
            filename,
            index=False,
            quoting=csv.QUOTE_ALL,
            escapechar='\\',
            lineterminator='\n'
        )

        print(f"✅ Saved CSV: {filename}")
        return filename


    def _build_batch_requests(self, df, include_fulltext_in_file=False, redaction_text='[FULL TEXT REMOVED]'):
        """
        Build batch request dicts from a DataFrame.

        - If `include_fulltext_in_file` is False, the Full Text in the generated
          prompt will be replaced by `redaction_text` (or omitted if None).
        - Returns a list of request dicts ready to be JSON-dumped into a .jsonl
          file for the Batch API.
        """
        batch_requests = []

        for i, row in df.iterrows():
            if include_fulltext_in_file:
                full_text_for_prompt = row.get('FullText', '')
            else:
                if redaction_text is None:
                    full_text_for_prompt = ''
                else:
                    full_text_for_prompt = redaction_text

            # Only include Full Text section if non-empty (avoids extra blank lines)
            full_text_section = f"Full Text: {full_text_for_prompt}" if full_text_for_prompt else ""

            system_prompt = f"""
You are an academic assistant. Given the title, abstract, and full text of a paper:

1. Generate a single concise research question.
2. Extract:
   - datasets used
   - models used
   - evaluation metrics used

Respond in JSON with:
         {{
         "research_question": "...",
         "GroundTruthDatasets": ["..."],
         "GroundTruthModels": ["..."],
         "GroundTruthMetrics": ["..."]
         }}

Only report the datasets, models, and metrics used in the experiments and not from the literature review or related work sections.
Each dataset, model and evaluation metric name must be composed from one to three words tops.
"""
            

            # Build the user-facing instructions (includes title, abstract, and full-text placeholder)
            user_prompt = f"""Return JSON with research_question, datasets, models, metrics. Format as valid JSON.

            Title: {row['Title']}
            Abstract: {row['Abstract']}
            {full_text_section}
            """.strip()

            batch_requests.append({
                "custom_id": f"paper_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.model_name,
                    "reasoning": {"effort": "medium"},
                    "input": user_prompt,
                    "instructions": system_prompt
                }
            })

        return batch_requests


    # ============================================================
    # Create Batch API JSONL File
    # ============================================================

    def save_batch_api_jsonl(self, df, filename=None):
        """
        Creates a JSONL file where each line is a batch request asking GPT to:
        - generate research question
        - extract ground truth metadata
        """

        # Save a redacted JSONL by default: full paper text is removed so the
        # saved file remains small. To upload the full text to the API, use
        # `submit_batch(..., df_for_upload=df, include_fulltext_in_upload=True)`
        batch_requests = self._build_batch_requests(df, include_fulltext_in_file=False)

        if filename is None:
            filename = os.path.join(self.batch_folder, "batch_api.jsonl")

        with open(filename, 'w', encoding='utf-8') as f:
            for item in batch_requests:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')

        print(f"✅ Saved Batch API JSONL: {filename}")
        return filename



    # ============================================================
    # Submit batch to OpenAI
    # ============================================================

    def submit_batch(self, jsonl_file=None, df_for_upload=None, include_fulltext_in_upload=False):
        """
        Upload and submit a batch. There are two modes:

        - Provide `jsonl_file`: upload that file and submit (backward compatible).
        - Provide `df_for_upload` and set `include_fulltext_in_upload=True`: a
          temporary JSONL containing full text will be created, uploaded, and
          removed afterwards (so you don't keep the large full-text file on disk).
        """

        temp_file = None

        if df_for_upload is not None and include_fulltext_in_upload:
            # Build a temporary upload file that contains full text for each prompt
            temp_file = os.path.join(self.batch_folder, "batch_api_upload_temp.jsonl")
            upload_requests = self._build_batch_requests(df_for_upload, include_fulltext_in_file=True)
            with open(temp_file, 'w', encoding='utf-8') as f:
                for item in upload_requests:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')

            jsonl_to_upload = temp_file
        elif jsonl_file:
            jsonl_to_upload = jsonl_file
        else:
            raise ValueError("Either provide `jsonl_file` or `df_for_upload` with `include_fulltext_in_upload=True`.")

        print("📤 Uploading batch JSONL to OpenAI...")
        uploaded = self.client.files.create(
            file=open(jsonl_to_upload, "rb"),
            purpose="batch"
        )

        print("🚀 Submitting batch...")
        batch = self.client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )

        print(f"🆔 Batch ID: {batch.id}")

        # Remove temporary upload file if we created it
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

        return batch.id



    # ============================================================
    # Download completed batch results
    # ============================================================

    def download_batch_results(self, batch_id, output_path="batch_output.jsonl"):
        print("📥 Checking batch status...")

        batch = self.client.batches.retrieve(batch_id)

        if batch.status != "completed":
            print(f"⏳ Batch not ready yet: {batch.status}")
            return None

        output_file_id = batch.output_file_id

        print("📥 Downloading results...")
        content = self.client.files.content(output_file_id)

        with open(output_path, "wb") as f:
            f.write(content.read())

        print(f"✅ Batch output saved to: {output_path}")
        return output_path



    # ============================================================
    # Parse batch output JSONL into a DataFrame
    # ============================================================

    def parse_batch_output(self, output_file):
        results = []

        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                custom_id = obj.get("custom_id")
                body = obj.get("response", {}).get("output_text")

                if not body:
                    continue

                try:
                    parsed = json.loads(body)
                except:
                    print(f"⚠️ JSON parsing error on item {custom_id}")
                    continue

                idx = int(custom_id.replace("paper_", ""))

                results.append((idx, parsed))

        return results



    # ============================================================
    # Download, parse, and save results
    # ============================================================

    def process_batch_results(self, batch_id, df_original, output_jsonl="batch_output.jsonl"):
        """
        Complete workflow:
        1. Download batch results
        2. Parse the JSONL output
        3. Merge with original DataFrame (title, abstract, year, etc.)
        4. Save final CSV with all metadata
        """

        # Step 1: Download
        print("\n📥 Downloading batch results...")
        output_path = self.download_batch_results(batch_id, output_jsonl)
        if not output_path:
            return None

        # Step 2: Parse
        print("📊 Parsing results...")
        results = self.parse_batch_output(output_path)
        
        if not results:
            print("⚠️ No results parsed from batch output.")
            return None

        # Step 3: Merge with original data
        print("🔗 Merging with original paper data...")
        results_dict = {idx: data for idx, data in results}
        
        df_result = df_original.copy()
        df_result['research_question'] = df_result.index.map(lambda i: results_dict.get(i, {}).get('research_question', ''))
        df_result['GroundTruthDatasets'] = df_result.index.map(lambda i: results_dict.get(i, {}).get('GroundTruthDatasets', []))
        df_result['GroundTruthModels'] = df_result.index.map(lambda i: results_dict.get(i, {}).get('GroundTruthModels', []))
        df_result['GroundTruthMetrics'] = df_result.index.map(lambda i: results_dict.get(i, {}).get('GroundTruthMetrics', []))

        # Remove FullText column before saving
        if 'FullText' in df_result.columns:
            df_result = df_result.drop(columns=['FullText'])

        # Convert lists to comma-separated strings for CSV
        for col in ['GroundTruthDatasets', 'GroundTruthModels', 'GroundTruthMetrics']:
            df_result[col] = df_result[col].apply(
                lambda val: ', '.join(val) if isinstance(val, list) and val else ''
            )

        # Step 4: Save CSV
        print("💾 Saving final CSV...")
        csv_file = self.save_csv(df_result)

        print(f"✅ Complete! Results saved to: {csv_file}")
        return csv_file



# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================

if __name__ == "__main__":

    API_KEY = os.getenv("OPENAI_API_KEY")

    batch_framework = BatchFramework(
        api_key=API_KEY,
        category="cs",
        years=[2025],
        papers_per_year=1000,
        model_name="gpt-5.1-2025-11-13",
        batch_folder="batches"
    )

    # ============================================================
    # PHASE 1: Fetch papers and submit batch
    # ============================================================
    print("\n" + "="*60)
    print("PHASE 1: Fetch papers and submit batch")
    print("="*60)

    # Step 1: Scrape papers (NO GPT CALLS)
    all_papers = []
    for year in batch_framework.years:
        papers = batch_framework.fetch_papers_from_arxiv(year)
        all_papers.extend(papers)

    df = pd.DataFrame(all_papers)

    if df.empty:
        print("❌ No papers fetched. Exiting.")
        exit(1)

    # Step 2: Create and save a redacted batch JSONL (no full text stored on disk)
    jsonl_file = batch_framework.save_batch_api_jsonl(df)

    # Step 3: Submit batch with full text in temporary upload
    batch_id = batch_framework.submit_batch(jsonl_file=None, df_for_upload=df, include_fulltext_in_upload=True)

    print("\n🚀 Batch submitted!")
    print(f"📌 Batch ID: {batch_id}")
    print("⏳ Wait 5-30 minutes for OpenAI to process, then run:")
    print(f"   batch_framework.process_batch_results('{batch_id}', df)")

