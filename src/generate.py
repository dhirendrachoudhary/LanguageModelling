import torch

def generate_text(model, start_text, char_to_idx, idx_to_char, device, length=100, model_type="lstm"):
    """Generate text from a trained LSTM model."""
    model.eval()
    input_seq = [char_to_idx[ch] for ch in start_text]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    hidden = (torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device),
              torch.zeros(model.lstm.num_layers, 1, model.lstm.hidden_size).to(device))

    generated_text = start_text
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            predicted_idx = torch.argmax(output, dim=1).item()
            generated_text += idx_to_char[predicted_idx]
            input_seq = torch.cat((input_seq[:, 1:], torch.tensor([[predicted_idx]]).to(device)), dim=1)

    return generated_text
