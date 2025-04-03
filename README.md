# **Language Modelling for Text Generation**

This project explores different architectures for text generation using deep learning, focusing on **LSTM** and **Transformer** models. The dataset consists of **fairy tales**, and the models generate new text based on learned patterns.  

---

## **📁 Project Structure**  

```
LanguageModelling/  
│── data/  
│   ├── fairy_tales.txt            # Input dataset  
│  
│── src/  
│   ├── __init__.py                # Module init  
│   ├── data_processing.py         # Text preprocessing & dataset handling  
│   ├── lstm_model.py              # LSTM-based text generation model  
│   ├── transformer_model.py       # Transformer-based text generation model  
│   ├── train.py                   # Training script supporting multiple architectures  
│   ├── generate.py                # Generate text from trained models  
│   ├── utils.py                   # Utility functions (saving/loading models, etc.)  
│  
│── experiments/  
│   ├── lstm_experiment_1/         # Logs, checkpoints, and results for LSTM model  
│   ├── transformer_experiment_1/  # Logs, checkpoints, and results for Transformer model  
│  
│── notebooks/                     # Jupyter notebooks for analysis  
│  
│── scripts/  
│   ├── train.sh                   # Shell script for training  
│   ├── generate_text.sh           # Shell script for text generation  
│  
│── tests/                         # Unit tests for models and preprocessing  
│  
│── main.py                        # Main entry point for training and text generation  
│── requirements.txt               # Python dependencies  
│── README.md                      # Project documentation  
```

---

## **Features**
- **Modular design** – Easily extend with new architectures  
- **Supports both LSTM and Transformer** for text generation  
- **Experiment tracking** – Logs, models, and generated text saved separately  
- **Flexible preprocessing** – Easy adaptation to new datasets  
- **Shell scripts for automation**  

---

## **Installation**
### **1️⃣ Set up environment**
```sh
python -m venv genai
source genai/bin/activate  # On Windows: genai\Scripts\activate
pip install -r requirements.txt
```
---
## **🚀 Usage**
### **🔹 Train a Model**
Modify `main.py` or `experiments/transformer_experiment_1/config.json` to select model type:  
```json
{
    "model_type": "transformer",  
    "epochs": 50,  
    "learning_rate": 0.001  
}
```
Then, run training:
```sh
bash scripts/train.sh
```
or manually:
```sh
python main.py
```

### **🔹 Generate Text**
After training, generate text using:
```sh
python main.py --generate
```
or using the script:
```sh
bash scripts/generate_text.sh
```

---

## **📈 Model Details**
### **1️⃣ LSTM Model**
- Uses an **embedding layer + LSTM layers + fully connected output**  
- Learns sequential patterns in text  
- Good for short sequences but struggles with long-term dependencies  

### **2️⃣ Transformer Model**
- Based on **attention mechanisms**  
- Processes entire sequences at once  
- More powerful for long-range dependencies  

---

## **📊 Results**
- **LSTM** performs well on small sequences, generating text in a coherent structure.  
- **Transformer** generalizes better for long sequences and generates more diverse text.  

---

## **🔧 Future Improvements**
🔹 Add support for **GPT-based** text generation  
🔹 Implement **character-level and word-level tokenization**  
🔹 Extend dataset with more diverse text sources  
🔹 Hyperparameter tuning and optimization  

---

## **💡 Contributions**
Feel free to **fork, improve, or report issues**! Contributions are welcome. 😊  

---

## **📜 License**
MIT License © 2025 Dhirendra Choudhary  

