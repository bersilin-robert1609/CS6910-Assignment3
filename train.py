from src.translator import Translator
import torch
import random

random.seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "embed_size": 16,
    "hidden_size": 512,
    "cell_type": "LSTM",
    "num_layers": 2,
    "dropout": 0.1,
    "learning_rate": 0.005,
    "optimizer": "SGD",
    "teacher_forcing_ratio": 0.5,
    "max_length": 50
}

model = Translator("tam", params, device)
model.train()

model.evaluate("hello")