import torch
import torch.nn as nn

def generate_text(model, start_text, char_to_idx, idx_to_char, device, length=100, model_type="lstm"):
    """Generate text from a trained language model (LSTM or Transformer)."""
    model.eval()
    input_seq = [char_to_idx[ch] for ch in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = start_text
    
    if model_type.lower() == "lstm":
        # Initialize hidden state for LSTM
        hidden = (torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device),
                 torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device))
        
        # Generate text using LSTM
        for _ in range(length):
            with torch.no_grad():
                output, hidden = model(input_seq, hidden)
                predicted_idx = torch.argmax(output, dim=1).item()
                generated_text += idx_to_char[predicted_idx]
                input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[predicted_idx]]).to(device)), dim=1)
    
    elif model_type.lower() == "transformer":
        # Get the maximum sequence length from the model's positional encoding
        max_seq_len = model.positional_encoding.size(1)
        
        # Generate text using Transformer
        for _ in range(length):
            with torch.no_grad():
                # Ensure input sequence doesn't exceed the maximum length
                if input_seq.size(1) >= max_seq_len:
                    input_seq = input_seq[:, -max_seq_len+1:]
                
                # The source is the full input sequence
                src = input_seq
                
                # The target is the source sequence minus the last token
                # This will make the model predict the next token
                tgt = input_seq[:, :-1]
                
                # Create a target mask to prevent looking ahead
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
                
                # Forward pass through the transformer
                output = model(src, tgt, src_mask=None, tgt_mask=tgt_mask)
                
                # Get the predicted index from the last position
                # output shape is [batch_size, vocab_size]
                predicted_idx = torch.argmax(output, dim=1).item()
                
                # Add the predicted character to the generated text
                generated_text += idx_to_char[predicted_idx]
                
                # Update the input sequence for the next iteration
                input_seq = torch.cat((input_seq, torch.tensor([[predicted_idx]]).to(device)), dim=1)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Use 'lstm' or 'transformer'.")

    return generated_text
