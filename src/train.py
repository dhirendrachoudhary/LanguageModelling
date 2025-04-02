import torch
import torch.optim as optim
import torch.nn as nn
from src.lstm_model import LSTMTextGenerator
from src.transformer_model import TransformerTextGenerator

def train_model(model, X_tensor, y_tensor, device, n_epochs=50, batch_size=16, lr=0.001, model_type="lstm"):
    """Train an LSTM or Transformer model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        hidden = None if model_type == "lstm" else None  # Hidden state only for LSTM

        for i in range(0, len(X_tensor) - batch_size, batch_size):
            inputs = X_tensor[i:i+batch_size].to(device)
            targets = y_tensor[i:i+batch_size].to(device)

            optimizer.zero_grad()

            if model_type == "lstm":
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden)
                outputs, hidden = model(inputs, hidden)
            else:
                tgt_input = inputs[:, :-1]  # Decoder input
                tgt_output = inputs[:, 1:]  # Expected output
                outputs = model(inputs, tgt_input)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    return model
