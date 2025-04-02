import torch
from src.data_processing import load_data, preprocess_text, create_sequences
from src.lstm_model import LSTMTextGenerator
from src.transformer_model import TransformerTextGenerator
from src.train import train_model
from src.generate import generate_text
from src.utils import save_model, load_model

# Load and process data
file_path = "data/fairy_tales.txt"
text = load_data(file_path)
encoded_text, char_to_idx, idx_to_char = preprocess_text(text)
vocab_size = len(char_to_idx)
seq_length = 10
X, y = create_sequences(encoded_text, seq_length)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Select model
MODEL_TYPE = "lstm"  # Change to "lstm" to train LSTM

embed_size = 16
hidden_size = 128
num_layers = 2
num_heads = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_TYPE == "lstm":
    model = LSTMTextGenerator(vocab_size, embed_size, hidden_size, num_layers).to(device)
else:
    model = TransformerTextGenerator(vocab_size, embed_size, num_heads, hidden_size, num_layers).to(device)

# Train model
model = train_model(model, X_tensor, y_tensor, device, model_type=MODEL_TYPE)

# Save trained model
experiment_folder = "experiments/transformer_experiment_1" if MODEL_TYPE == "transformer" else "experiments/lstm_experiment_1"
save_model(model, f"{experiment_folder}/model_checkpoint.pth")

# Generate text
start_text = "once upon a"
print(generate_text(model, start_text, char_to_idx, idx_to_char, device, model_type=MODEL_TYPE))