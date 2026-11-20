import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class RULPredictorLSTM(nn.Module):

    def __init__(
        self, input_dim: int = 4, hidden_dim: int = 64, num_layers: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_rul_model(
    X_train, y_train, epochs: int = 40, batch_size: int = 32, seed: int = 42
):
    torch.manual_seed(seed)

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    num_features = int(X_train.shape[-1])
    model = RULPredictorLSTM(input_dim=num_features)
    criterion = nn.HuberLoss(delta=10.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model
