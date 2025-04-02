import torch

def save_model(model, filepath="model.pth"):
    """Save trained model."""
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath, vocab_size, embed_size, hidden_size, num_layers, device):
    """Load trained model."""
    model = model_class(vocab_size, embed_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    return model