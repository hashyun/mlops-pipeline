from pathlib import Path
import pandas as pd
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ImportError as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    MinMaxScaler = None
    mean_absolute_error = mean_squared_error = None
    raise ImportError("Required dependencies for the LSTM pipeline are missing") from exc


class LSTM(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size=32, num_layers=2, output_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
            torch.zeros(self.num_layers, self.seq_len, self.hidden_size),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def run_lstm_pipeline():
    data_dir = Path('notebook/data') if Path('notebook/data').is_dir() else Path('data')
    train = pd.read_csv(data_dir / 'merged_train (1).csv')
    test = pd.read_csv(data_dir / 'merged_test (1).csv')

    train.columns = [c.lower().strip() for c in train.columns]
    test.columns = [c.lower().strip() for c in test.columns]

    train['date'] = pd.to_datetime(train['date'])
    test['date'] = pd.to_datetime(test['date'])

    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    seq_length = 5
    pred_size = 6
    train = data[:-pred_size].copy()
    test = data[-(pred_size + seq_length):].copy()

    target_col = 'close'
    feature_cols = ['neu', 'trend', 'close']

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train[feature_cols])
    scaled_test = scaler.transform(test[feature_cols])

    close_scaler = MinMaxScaler()
    scaled_close_train = close_scaler.fit_transform(train[[target_col]]).flatten()
    scaled_close_test = close_scaler.transform(test[[target_col]]).flatten()

    def create_sequences(X, y, seq_length):
        xs, ys = [], []
        for i in range(len(X) - seq_length):
            xs.append(X[i:i + seq_length])
            ys.append(y[i + seq_length])
        return np.array(xs), np.array(ys)

    X_train, y_train = create_sequences(scaled_train, scaled_close_train, seq_length)
    X_test, y_test = create_sequences(scaled_test, scaled_close_test, seq_length)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=len(feature_cols), seq_len=seq_length).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-5)
    criterion = nn.MSELoss()

    num_epochs = 5
    for epoch in range(1, num_epochs + 1):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            model.reset_hidden_state()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        model.reset_hidden_state()
        preds = model(X_test.to(device)).cpu().numpy().flatten()

    preds_inv = close_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    y_test_inv = close_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = mean_squared_error(y_test_inv, preds_inv)
    print(f"Validation MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    out_path = data_dir / 'lstm_predictions.csv'
    pd.DataFrame({'date': test['date'].values[seq_length:], 'pred': preds_inv}).to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == '__main__':
    run_lstm_pipeline()
