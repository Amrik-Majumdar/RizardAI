# RizardAI Emotion Classification Model

Lightweight emotion detection system designed for real-time chat analysis and browser-friendly deployment. The core model is trained on Google’s **GoEmotions** dataset and powers emotion-aware response generation in the RizardAI stack.

---

## Overview

RizardAI detects emotional tone in short text messages and uses that signal to guide downstream LLM responses. The system is optimized for:

* Fast CPU inference
* Low memory footprint
* Client-side or edge-friendly deployment
* Chat and messaging use cases

The model maps GoEmotions’ 27 fine-grained emotions into **7 practical classes** suitable for conversational UX.

**Emotion classes:**

* neutral
* happy
* sad
* angry
* fear
* surprise
* disgust

---

## Dataset

**GoEmotions (Google Research)**

* ~58,000 Reddit comments
* 27 emotion labels + neutral
* Single-label samples only are used
* Labels are remapped into 7 broader categories for stability and deployment efficiency

Only samples with exactly one emotion label are included to avoid ambiguity.

---

## Model Architecture

**Backbone:** `distilbert-base-uncased`

**Why DistilBERT:**

* ~40% faster than BERT
* ~60% smaller
* Strong performance on short text
* Practical for CPU-only environments

**Classifier Head:**

* Dropout (0.3)
* Linear projection to 7 classes

Total parameters: ~66M

---

## Training Setup

### Environment

Python 3.9+

Key dependencies:

* torch
* transformers
* datasets
* scikit-learn
* numpy, pandas
* matplotlib, seaborn

All dependencies are installed programmatically in the setup block.

---

### Configuration

* Max sequence length: **64 tokens**
* Batch size: **64 (train)** / **128 (val)**
* Epochs: **3**
* Optimizer: **AdamW**
* Learning rate: **3e-5**
* Scheduler: **Linear warmup + decay**
* Loss: **CrossEntropyLoss** with class weighting and label smoothing

Training is fully reproducible via fixed random seeds.

---

## Training Results

**Best validation performance:**

* Accuracy: **0.6608**
* Macro F1: **0.5598**
* Weighted F1: **0.6728**

### Per-class F1 (validation)

* neutral: 0.656
* happy: 0.804
* sad: 0.561
* angry: 0.520
* fear: 0.559
* surprise: 0.379
* disgust: 0.440

Happy and neutral perform strongest; surprise and disgust are the most challenging due to class imbalance and semantic overlap.

---

## Evaluation & Analysis

The training pipeline produces:

* Loss, accuracy, and F1 curves
* Confusion matrices (raw + normalized)
* Per-class precision/recall/F1
* Confidence and entropy analysis
* Calibration plots
* ROC and Precision–Recall curves (multi-class)

These diagnostics are saved as high-resolution figures for inspection.

---

## Saved Artifacts

* `best_emotion_model.pth`

  * Model weights
  * Training configuration (model name, max length, number of classes)

This checkpoint is used directly by the deployment module.

---

## Deployment Module

The deployment layer wraps the emotion model in a lightweight Python class used alongside an LLM.

### Key Features

* Loads the trained DistilBERT emotion model
* Produces:

  * Primary emotion
  * Confidence score
  * Entropy-based uncertainty
  * Full emotion probability distribution
* Applies heuristic adjustments for flirty / romantic language
* Visualizes emotion distributions
* Feeds emotion context into an external LLM (Cerebras API)

---

## Emotion-Aware Response Generation

RizardAI combines:

1. **Statistical emotion detection** (DistilBERT)
2. **Keyword-based intent boosting** (flirty / romantic signals)
3. **LLM prompting** with emotion, confidence, and vibe controls

The LLM generates three reply styles:

* SAFE: low-risk, warm
* BOLD: confident, forward
* PLAYFUL: teasing, flirty

All outputs are actual text messages, not explanations.

---

## Intended Use

* Chat emotion analysis
* Dating and messaging assistants
* Conversational UX tuning
* Real-time sentiment-aware response systems

Not intended for mental health diagnosis or high-stakes emotional inference.

---

## Notes

* Model is CPU-friendly and browser-deployable with further conversion (ONNX / WebGPU).
* Emotion classes are intentionally broad to improve robustness in short, informal text.
* Keyword heuristics are layered on top of model output by design.

---

## License & Credits

* GoEmotions dataset: Google Research
* Transformers: Hugging Face
* Model architecture: DistilBERT

RizardAI is an experimental system built for practical conversational intelligence, not academic benchmarking.
