import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 5):
        self.seq_len = seq_len
        self.y = df['Close'].values.astype(np.float32)
        self.X = df.drop(columns=['Close']).values.astype(np.float32)

    def __len__(self):
        return len(self.y) - self.seq_len

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return torch.tensor(X_seq), torch.tensor(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

def train_lstm(train_df: pd.DataFrame, epochs: int = 30, seq_len: int = 5) -> LSTMModel:
    dataset = StockDataset(train_df, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = LSTMModel(input_size=train_df.shape[1] - 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
    return model

def generate_predictions(model: LSTMModel, train_df: pd.DataFrame, test_df: pd.DataFrame, seq_len: int = 5):
    model.eval()
    history = train_df.drop(columns=['Close']).values.astype(np.float32)
    y_history = train_df['Close'].values.astype(np.float32)
    test_X = test_df.drop(columns=['Close']).values.astype(np.float32)
    preds = []
    with torch.no_grad():
        for i in range(len(test_df)):
            start = max(0, len(history) - seq_len)
            window = history[start:]
            if len(window) < seq_len:
                pad = np.repeat(window[0:1], seq_len - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            x = torch.tensor(window[-seq_len:]).unsqueeze(0)
            pred = model(x).item()
            preds.append(pred)
            history = np.vstack([history, test_X[i]])
            y_history = np.append(y_history, pred)
    return preds


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def save_model(model: LSTMModel, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, input_size: int) -> LSTMModel:
    model = LSTMModel(input_size)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

