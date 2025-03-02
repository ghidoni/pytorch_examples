import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

x_train = torch.randint(0, 1000, (100000, 10, 100)).float()
x_test = torch.randint(0, 1000, (10000, 10, 100)).float()
y_train = torch.randint(0, 2, (100000,)).float()
y_test = torch.randint(0, 2, (10000,)).float()


class CustomDataset(Dataset):
    def __init__(self, x, y, reverse_sequence=False):
        self.x = x
        self.y = y
        self.reverse_sequence = reverse_sequence

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.reverse_sequence:
            x = torch.flip(x, dims=[0])  # Reverse the sequence
        return x, y


train_dataset = CustomDataset(x_train, y_train, reverse_sequence=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(x_test, y_test, reverse_sequence=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class LSTMTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.2,
        use_normalization=True,
    ):
        super(LSTMTimeSeriesClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_normalization = use_normalization

        if self.use_normalization:
            self.norm = nn.LayerNorm(self.input_size)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if self.use_normalization:
            x = self.norm(x)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the output of the last time step
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10,
    lr=0.001,
    max_batches=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_accuracy = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        batch_count = 0

        # Limit batches if specified
        if max_batches:
            train_iter = iter(train_loader)
            train_batches = min(max_batches, len(train_loader))
            batch_data = [(next(train_iter), i) for i in range(train_batches)]
        else:
            batch_data = [(batch, i) for i, batch in enumerate(train_loader)]

        progress_bar = tqdm(
            batch_data, desc=f"Epoch {epoch+1} batches", leave=True
        )

        for (x, y), i in progress_bar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

            progress_bar.set_postfix(
                {
                    "Batch": f"{i+1}/{len(batch_data)}",
                    "Loss": loss.item(),
                }
            )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/batch_count:.4f}"
        )

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_batch_count = 0

        # Limit validation batches if specified
        if max_batches:
            test_iter = iter(test_loader)
            test_batches = min(max_batches, len(test_loader))
            test_data = [(next(test_iter), i) for i in range(test_batches)]
        else:
            test_data = [(batch, i) for i, batch in enumerate(test_loader)]

        val_progress_bar = tqdm(
            test_data, desc=f"Epoch {epoch+1} validation", leave=True
        )

        with torch.no_grad():
            for (x, y), i in val_progress_bar:
                x, y = x.to(device), y.to(device)
                y_pred = model(x).squeeze(-1)
                batch_loss = criterion(y_pred, y).item()
                val_loss += batch_loss
                predicted = (y_pred > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()
                val_batch_count += 1

                val_progress_bar.set_postfix(
                    {
                        "Batch": f"{i+1}/{len(test_data)}",
                        "Loss": batch_loss,
                    }
                )

        accuracy = correct / total
        avg_val_loss = val_loss / val_batch_count

        print(
            f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pt")
            print(f"✓ New best model saved with accuracy: {best_accuracy:.4f}")

    print(f"Best test accuracy: {best_accuracy:.4f}")
    return model


model = LSTMTimeSeriesClassifier(
    input_size=100,
    hidden_size=128,
    num_layers=2,
    num_classes=2,
    dropout=0.2,
)

model = train_model(
    model,
    train_dataloader,
    test_dataloader,
    num_epochs=10,
    lr=0.001,
    max_batches=1000,
)


def evaluate_model(model, test_loader):
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze(-1)
            predicted = (y_pred > 0.5).float()

            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    accuracy = correct / total

    # More metrics calculation can be added here (AUC, F1, etc.)
    return accuracy, all_preds, all_targets
