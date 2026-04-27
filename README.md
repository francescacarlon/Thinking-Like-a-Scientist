# Thinking Like a Scientist? A Structural Study of LLM-Generated Research Methods

Research code and artifacts for analyzing how large language models suggest and classify the core ingredients of AI research papers: datasets, models, and evaluation metrics.

This repository contains:

- the intermediate data used in the study
- scripts for batch extraction, suggestion generation, normalization, and classification
- figure and table generation code
- precomputed figures and tables ready for paper use

## Project Overview

The workflow in this repository compares a reference inventory extracted from a corpus of LLM-related arXiv papers against suggestions produced by multiple LLMs. The analysis focuses on:

- entity-level differences in datasets, models, and metrics
- taxonomy-level differences such as provider, openness, task type, and evaluation type
- co-occurrence structure between datasets, models, and metrics
- robustness and validation checks

The repository is organized as a research artifact rather than as a packaged Python library. Most scripts are intended to be run directly.

## Repository Structure

```text
.
|-- data/                  # Intermediate and processed data artifacts
|-- figures/               # Generated figure outputs
|-- tables/                # Generated table outputs
|-- src/
|   |-- pipeline/          # Data collection, extraction, normalization, classification
|   `-- figures/           # Figure and table generation scripts
|-- .env.example.example   # Example environment variables
`-- .gitignore
```

Key folders:

- `src/pipeline/`: scripts for corpus construction, batch prompting, normalization, ablations, and validation helpers
- `src/figures/`: scripts that generate publication-ready figures and tables
- `data/`: tracked inputs and derived analysis outputs
- `figures/` and `tables/`: exported artifacts already produced by the analysis

## Main Workflow

At a high level, the project follows this pipeline:

1. Build a paper corpus from arXiv and extract full text.
2. Use batch LLM calls to extract a reference inventory and generate research-question-based suggestions.
3. Normalize entity names with deterministic and fuzzy matching.
4. Classify datasets, models, and metrics into a fixed taxonomy.
5. Compare reference vs. LLM outputs and generate paper figures and tables.

Representative scripts:

- `src/pipeline/01_batch_gt_extraction.py`
- `src/pipeline/02_batch_llm_suggestions.py`
- `src/pipeline/03_normalize_entities.py`
- `src/pipeline/04_compare_gt_llms.py`
- `src/pipeline/05_batch_classification.py`
- `src/figures/generate_tables.py`

## Environment Setup

Create a virtual environment and install the libraries used by the scripts:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scipy thefuzz python-dotenv openai requests pymupdf arxiv openpyxl
```

Some scripts also rely on:

- a working Git installation
- optional LaTeX, if you want matplotlib text rendering or paper-oriented exports to match the original setup

## Environment Variables

Copy the example file and add your API keys if you plan to rerun the LLM pipeline:

```bash
copy .env.example.example .env
```

Expected variables:

```env
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

Note:

- OpenAI is used by the GPT-based batch scripts.
- Google is used by the Gemini-related workflow.
- Some auxiliary scripts for other providers may require additional credentials or local adjustments.

## Reproducing Outputs

### Figures and Tables From Existing Artifacts

If you want to regenerate figures and tables from the tracked outputs already in `data/`, start with the scripts in `src/figures/`.

Examples:

```bash
cd src\figures
python generate_tables.py
python figure_overview.py
python figure_provider_composite.py
python figure_robustness_dashboard.py
```

By default, figure and table scripts write to repository output folders defined in `src/figures/paths.py`.

Optional output overrides:

- `FIGURES_OUT_DIR`
- `PAPER_FIGURES_OUT_DIR`
- `TABLES_OUT_DIR`
- `PAPER_TABLES_OUT_DIR`
- `OUTPUTS_OUT_DIR`

### Full Pipeline Reruns

The full LLM pipeline is available under `src/pipeline/`, but it is not yet standardized as a single command-line entry point. Several scripts use research-specific file paths and assumptions from the original project workflow.

If you want to rerun the full pipeline, review each script before execution and verify:

- input and output paths
- provider/model names
- API credentials
- expected batch file locations under `data/batches/`

## Included Outputs

This repository already includes many generated artifacts, including:

- publication figures in `figures/`
- publication tables in `tables/`
- normalized count summaries in `data/analysis_outputs_names/`
- classification and ablation outputs in `data/batches/batch_suggestions_results/`

This makes the repository usable both as:

- a reproducibility package for the paper outputs
- a codebase for extending the analysis

## Reproducibility Notes

- The codebase is script-driven and reflects an active research workflow.
- Some scripts contain hard-coded paths or provider-specific assumptions.
- Precomputed outputs are tracked in the repository to make the main results inspectable without rerunning expensive API jobs.

## Suggested Citation

If this repository supports a paper, add the final citation here once the manuscript details are fixed.

```bibtex
@misc{thinking_like_a_scientist,
  title  = {Thinking Like a Scientist},
  author = {Author(s) to be added},
  year   = {2026}
}
```

## License

Add the project license here, for example `MIT`, `Apache-2.0`, or a research-specific license once decided.
