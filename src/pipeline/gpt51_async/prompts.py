SYSTEM_PROMPT = f"""
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
- Kernel 
- Probabilistic

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
