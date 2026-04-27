SYSTEM_PROMPT = """
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
"gemini_suggested_dataset": ["..."],
"gemini_suggested_model": ["..."],
"gemini_suggested_evaluation_metric": ["..."],
"gemini_suggested_pipeline": "..."
}}
"""

USER_PROMPT = f"""Return JSON categorization objects for each of the following fields:
gemini_suggested_dataset, gemini_suggested_model, gemini_suggested_evaluation_metric, gemini_suggested_pipeline
"""
