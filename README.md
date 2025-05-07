# Language Modelling for Fairy Tale Generation

## Table of Contents

* [Introduction](#introduction)
* [Project Motivation](#project-motivation)
* [The Challenge: Fairy Tale Generation](#the-challenge-fairy-tale-generation)
* [Models Under Investigation](#models-under-investigation)
* [Dataset and Preprocessing](#dataset-and-preprocessing)
* [Methodology](#methodology)
    * [Character-Level Approach](#character-level-approach)
    * [LSTM Model](#lstm-model)
    * [Transformer Model](#transformer-model)
    * [Training Process](#training-process)
* [Getting Started & Running the Models](#getting-started--running-the-models)
    * [Prerequisites](#prerequisites)
    * [Setup](#setup)
    * [Training Models](#training-models)
    * [Generating Text](#generating-text)
* [Results and Evaluation](#results-and-evaluation)
    * [Loss Curves](#loss-curves)
    * [Qualitative Analysis](#qualitative-analysis)
    * [Quantitative Metrics](#quantitative-metrics)
    * [Generated Fairy Tale Samples](#generated-fairy-tale-samples)
* [Model Comparison](#model-comparison)
* [Repository Structure](#repository-structure-suggestion)
* [Contributing](#contributing)
* [License](#license)

---

## Introduction

Natural Language Generation (NLG) is a rapidly evolving field in AI and NLP, focusing on creating understandable and contextually relevant human language text. At the core of modern NLG lies **Language Modelling**, which learns the probability distribution of token sequences (words, characters, etc.). This project delves into creative text generation by training language models to produce fairy tales.

### Project Motivation

This project is driven by the desire to explore the capabilities of modern neural network architectures (LSTM and Transformer) in the creative domain of fairy tale generation. The goal is to generate text that is not only grammatically correct and coherent but also creative, stylistically consistent, and engaging.

### The Challenge: Fairy Tale Generation

Fairy tales were chosen for their:
* **Distinct Style:** Unique vocabulary, sentence structures, and narrative voice.
* **Narrative Structure:** Recognizable plot patterns, archetypal characters, and moral lessons.
* **World Knowledge:** Implicit understanding of common tropes.
* **Coherence:** Maintaining plot, character, and setting consistency.

### Models Under Investigation

This project implements, trains, and compares two leading neural network architectures for sequence modelling:
1.  **Long Short-Term Memory (LSTM):** A type of RNN adept at handling sequential data and capturing long-range dependencies.
2.  **Transformer:** A newer architecture relying entirely on self-attention mechanisms, known for state-of-the-art performance in NLP.

The comparison aims to shed light on their respective strengths and weaknesses in generating creative, structured text using character-level inputs.

---

## Dataset and Preprocessing

* **Data Source:** A corpus composed primarily of text from classic fairy tales.
* **Tokenization Strategy:** Character-Level Deep Dive.
    * Every character (letters, digits, punctuation, whitespace) is treated as an individual token.
    * Results in a small vocabulary but requires models to handle much longer sequences.
    * Advantages: No Out-of-Vocabulary (OOV) issues, implicit morphology learning.
    * Disadvantages: Longer sequences, increased context requirement, potentially weaker semantic units.
* **Vocabulary Construction:**
    * Unique characters mapped to integer IDs.
    * Special tokens added: `<|pad|>`, `<|unk|>`, `<|endoftext|>`.
    * Vocabulary saved (e.g., `vocab.json`) for consistent use.
* **Data Cleaning:** (Implicit/explicit steps like lowercasing, whitespace normalization).
* **Data Splitting:** Training and Validation sets (e.g., 80-90% train, 10-20% validation).

---

## Methodology

### Character-Level Approach
The project employs character-level tokenization. This means the models learn to predict the next character in a sequence based on the preceding characters.

### Foundational Concepts
* **Embeddings:** Character IDs are converted into dense vector representations (Embedding Layer).
* **Autoregressive Prediction:** The model predicts the next token based on previous tokens.

### LSTM Model
* **Architecture:** Processes sequences step-by-step, using gates (Forget, Input, Output) to control information flow through a cell state. Can be stacked in multiple layers.
* **Output:** Final hidden states are passed through a linear layer to predict the next character.

### Transformer Model
* **Architecture:** Relies on self-attention mechanisms, processing the entire sequence simultaneously. Key components include:
    * Multi-Head Self-Attention (Queries, Keys, Values)
    * Positional Encodings
    * Causal Masking (for autoregressive generation)
    * Position-wise Feedforward Networks
    * Residual Connections & Layer Normalization
    * Can be stacked in multiple encoder blocks.
* **Output:** Final representations are passed through a linear layer for character prediction.

### Training Process
1.  **Initialization:** Load model, tokenizer, datasets, optimizer (Adam), loss function (CrossEntropyLoss).
2.  **Epoch Iteration:** Loop through the training dataset multiple times.
3.  **Batch Processing:**
    * Forward pass to get logits.
    * Calculate loss.
4.  **Backpropagation:** Compute gradients.
5.  **Optimization:** Update model parameters.
6.  **Gradient Clipping:** Prevent exploding gradients (max_norm: 1.0).
7.  **Validation:** Evaluate on the validation set after each epoch.
8.  **Checkpointing:** Save the best model based on validation loss (e.g., `best_model.pt`).

#### Key Hyperparameters Investigated:
* Learning Rate (e.g., LSTM: 0.001, Transformer: 0.0001)
* Embedding Dimension (e.g., 256)
* Hidden/Feedforward Dimensions (e.g., LSTM Hidden: 512, Transformer FF: 512)
* Number of Layers (e.g., 2)
* Dropout Rate (e.g., LSTM: 0.2, Transformer: 0.1)
* Sequence Length (e.g., 50, 60)
* Batch Size (e.g., 6, 64)

---

## Getting Started & Running the Models

This section guides you through setting up the environment and running the models.

### Prerequisites
* Python 3.10
* PyTorch 1.12.1
* NumPy 1.22.4
* Matplotlib 3.5.2
* TQDM 4.67.1
* Other dependencies (as listed in your report, or from a `requirements.txt` file):
    ```
    contourpy==1.3.0
    cycler==0.12.1
    fonttools==4.57.0
    importlib-resources==6.5.2
    kiwisolver==1.4.7
    packaging==24.2
    pillow==11.1.0
    pyparsing==3.2.3
    python-dateutil==2.9.0.post0
    six==1.17.0
    typing-extensions==4.13.1
    zipp==3.21.0
    ```

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/dhirendrachoudhary/LanguageModelling.git](https://github.com/dhirendrachoudhary/LanguageModelling.git)
    cd LanguageModelling
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt # Or pip install torch torchvision torchaudio numpy matplotlib tqdm ...
    ```
4.  **Download/Prepare Dataset:**
        ```bash
        # Example:
        # python preprocess_data.py --input_file path/to/raw_fairy_tales.txt --output_dir data/
        ```

### Training Models
    ```bash
    # Example for training LSTM:
    # python train.py --model_type lstm --config_path config/lstm_config.json --data_path data/processed_text.txt --vocab_path data/vocab.json

    # Example for training Transformer:
    # python train.py --model_type transformer --config_path config/transformer_config.json --data_path data/processed_text.txt --vocab_path data/vocab.json
    ```

### Generating Text
    ```bash
    # Example for text generation:
    # python generate.py --model_path checkpoints/best_lstm_model.pt --vocab_path data/vocab.json --seed_text "Once upon a time" --max_length 200 --sampling_strategy top-k --k 10
    ```
* **Sampling Strategies Available:**
    * Greedy Search
    * Top-k Sampling (specify `k`)
    * Nucleus (Top-p) Sampling (specify `p`)
    * Temperature Scaling (specify `temperature`)

---

## Results and Evaluation

### Loss Curves
Training and validation loss curves were monitored to track learning dynamics.
* The LSTM model showed steady decline in training loss, with validation loss closely tracking.
* The Transformer model converged faster initially but showed a slight gap later, hinting at mild overfitting.

*Interactive Suggestion:*
    ```markdown
    ![Loss Curves](path/to/your/loss_curve_image.png)
    ```

### Qualitative Analysis
* **LSTM:** Produced readable character sequences, some fairy tale elements. Struggled with global coherence and showed repetition.
* **Transformer:** Consistently produced more fluent and longer coherent passages. Exhibited higher lexical diversity and better captured fairy tale style, including imaginative elements.
  

### Quantitative Metrics
| Feature          | LSTM Model         | Transformer Model  | Notes                                |
|------------------|--------------------|--------------------|--------------------------------------|
| Training Time    | ~2 hrs / 15 mins/ep| ~1.5 hrs / 11 mins/ep| Tested on 1× V100 GPU (Report: RTX 2080) |
| Best Valid Loss  | 1.45               | 1.38               | Cross-entropy loss                   |
| Best Valid PPL   | 4.26               | 3.97               | exp(loss), Lower is better           |
| Distinct-1 (Gen.)| 0.08               | 0.10               | Based on ~1K sampled tokens          |
| Distinct-2 (Gen.)| 0.25               | 0.35               | Transformer: more diverse phrases    |
| BLEU Score (Gen.)| 10.5               | 12.0               | Ref: curated fairy tales corpus      |

