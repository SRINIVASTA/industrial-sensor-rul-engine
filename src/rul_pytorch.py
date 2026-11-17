import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class RULPredictorLSTM(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

def train_rul_model(X_train, y_train, epochs: int = 25, batch_size: int = 32):
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = RULPredictorLSTM(input_dim=X_train.shape[2])
    criterion = nn.HuberLoss()  # Robust to sensor outliers
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    return model
