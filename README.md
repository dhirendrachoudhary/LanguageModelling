# Language Modeling with LSTM and Transformer Architectures

This project implements a flexible text generation system using both LSTM (Long Short-Term Memory) and Transformer neural network architectures. It allows you to train language models on text data and generate new text based on a starting prompt.

## Project Description

The language modeling system supports character-level text generation with two different neural network architectures:

1. **LSTM Model**: Uses recurrent neural networks with LSTM cells to model sequential data
2. **Transformer Model**: Uses self-attention mechanisms to capture relationships between characters

Both models can be trained on custom text data and used to generate coherent text that mimics the style and patterns of the training data.

## Project Structure

```
LanguageModelling/
├── data/                   # Directory for training data files
│   └── fairy_tales.txt     # Sample training text file
├── experiments/            # Directory for saving model checkpoints
│   ├── lstm_experiment_1/  # LSTM model checkpoints
│   └── transformer_experiment_1/ # Transformer model checkpoints
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing utilities
│   ├── generate.py         # Text generation functionality
│   ├── lstm_model.py       # LSTM model architecture
│   ├── train.py            # Model training functionality
│   ├── transformer_model.py # Transformer model architecture
│   └── utils.py            # Utility functions for saving/loading models
├── main.py                 # Main script for training and generating text
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/LanguageModelling.git
   cd LanguageModelling
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install torch numpy argparse
   ```

## Usage

### Training a Model

To train a language model, use the `main.py` script with the `train` mode:

```bash
# Train an LSTM model
python main.py --mode train --model_type lstm

# Train a Transformer model
python main.py --mode train --model_type transformer
```

The trained model will be saved in the `experiments/` directory.

### Generating Text

To generate text using a trained model:

```bash
# Generate text using an LSTM model
python main.py --mode generate --model_type lstm --model_path experiments/lstm_experiment_1/model_checkpoint.pth --start_text "once upon a"

# Generate text using a Transformer model
python main.py --mode generate --model_type transformer --model_path experiments/transformer_experiment_1/model_checkpoint.pth --start_text "once upon a"
```

## Model Architectures

### LSTM Model

The LSTM model architecture consists of:
- An embedding layer that converts character indices to dense vectors
- LSTM layers that process the sequential data
- A fully connected output layer that produces predictions for the next character

```python
class LSTMTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
```

### Transformer Model

The Transformer model uses the architecture from the "Attention is All You Need" paper:
- An embedding layer for character vectors
- Positional encoding to capture sequence order
- Multi-head self-attention layers
- Feed-forward networks
- A final output layer for next-character prediction

```python
class TransformerTextGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers):
        super(TransformerTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_size))
        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(embed_size, vocab_size)
```

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- (Optional) CUDA-capable GPU for faster training

## Example Commands

### Data Preparation

The system works with plain text files. You can place your own text data in the `data/` directory.

### Training

Train an LSTM model with default parameters:
```bash
python main.py --mode train --model_type lstm
```

Train a Transformer model with custom parameters:
```bash
python main.py --mode train --model_type transformer
```

### Text Generation

Generate text with an LSTM model:
```bash
python main.py --mode generate --model_type lstm --model_path experiments/lstm_experiment_1/model_checkpoint.pth --start_text "once upon a time"
```

Generate text with a Transformer model:
```bash
python main.py --mode generate --model_type transformer --model_path experiments/transformer_experiment_1/model_checkpoint.pth --start_text "the king and queen"
```

## Advanced Configuration

You can modify the hyperparameters in `main.py` to customize the models:

```python
# Model hyperparameters
embed_size = 16     # Size of character embeddings
hidden_size = 128   # Size of hidden layers
num_layers = 2      # Number of LSTM or Transformer layers
num_heads = 4       # Number of attention heads (Transformer only)
```

## License

[MIT License](LICENSE)
This project is licensed under the MIT License. See the LICENSE file for details.
This project is open-source and free to use, modify, and distribute under the terms of the MIT License.
Feel free to contribute to the project by submitting issues or pull requests.

## Contributors
[Dhirendra Choudhary]