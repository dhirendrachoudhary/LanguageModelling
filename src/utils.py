import torch

def save_model(model, filepath="model.pth"):
    """Save trained model."""
    torch.save(model.state_dict(), filepath)

def load_model(model_class, filepath, vocab_size, embed_size, hidden_size, num_layers, device, num_heads=None):
    """Load trained model.
    
    Args:
        model_class: The model class (LSTMTextGenerator or TransformerTextGenerator)
        filepath: Path to the saved model file
        vocab_size: Size of the vocabulary
        embed_size: Embedding dimension
        hidden_size: Hidden size (for LSTM) or hidden_dim (for Transformer)
        num_layers: Number of layers
        device: Device to load the model onto
        num_heads: Number of attention heads (only for Transformer)
    
    Returns:
        Loaded model
    """
    from src.lstm_model import LSTMTextGenerator
    from src.transformer_model import TransformerTextGenerator
    
    if model_class == LSTMTextGenerator:
        # For LSTM models
        model = model_class(vocab_size, embed_size, hidden_size, num_layers)
    elif model_class == TransformerTextGenerator:
        # For Transformer models
        if num_heads is None:
            raise ValueError("num_heads parameter is required for Transformer models")
        model = model_class(vocab_size, embed_size, num_heads, hidden_size, num_layers)
    else:
        raise ValueError(f"Unsupported model class: {model_class.__name__}")
    
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    return model
