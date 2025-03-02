import torch
import torch.nn as nn
import math

# import h5py
# import numpy as np
# import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# TODO: Implement RNNTimeSeriesClassifier class
# TODO: Test difference between sinusoidal and linear positional encoding
# TODO: Find new possible positional encoding methods
# TODO: Test differences between raw input and normalized input
# TODO: Test different loss functions, eg. CrossEntropyLoss, BCELoss, etc.

# with h5py.File("./x_train.h5", "r") as f:
#     x_train = f["tensor"][:]
# x_train = np.nan_to_num(x_train, nan=-1)
# x_train = torch.from_numpy(x_train).float()

# with h5py.File("./x_test.h5", "r") as f:
#     x_test = f["tensor"][:]
# x_test = np.nan_to_num(x_test, nan=-1)
# x_test = torch.from_numpy(x_test).float()

# y_train = pd.read_parquet("./y_train.pq")
# y_test = pd.read_parquet("./y_train.pq")

x_train = torch.randint(0, 1000, (100000, 10, 100)).float()
x_test = torch.randint(0, 1000, (10000, 10, 100)).float()
y_train = torch.randint(0, 2, (100000,)).float()
y_test = torch.randint(0, 2, (10000,)).float()


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


train_dataset = CustomDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(x_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CustomPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, encoding_type="learnable"):
        super(CustomPositionalEncoding, self).__init__()

        # Let's define the encoding matrix based on the desired encoding type
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            # Sinusoidal Encoding
            self.register_buffer(
                "positional_encoding",
                self._get_sinusoidal_encoding(embed_dim, max_len),
            )
        elif encoding_type == "linear":
            # Linear Encoding (values increase linearly with position)
            self.register_buffer(
                "positional_encoding",
                self._get_linear_encoding(embed_dim, max_len),
            )
        elif encoding_type == "learnable":
            # Learnable Encoding (the model learns the encoding)
            self.positional_encoding = nn.Parameter(
                torch.zeros(max_len, embed_dim)
            )
            nn.init.xavier_uniform_(
                self.positional_encoding
            )  # Initialize the learnable parameters

    def forward(self, x):
        # Ensure the pos encoding matrix aligns with the input's seq length
        return x + self.positional_encoding[: x.size(1), :].unsqueeze(0)

    def _get_sinusoidal_encoding(self, embed_dim, max_len):
        # Generates a sinusoidal positional encoding matrix
        pos_encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def _get_linear_encoding(self, embed_dim, max_len):
        # Generates a linear positional encoding matrix
        return (
            torch.linspace(0, 1, steps=max_len)
            .unsqueeze(1)
            .expand(max_len, embed_dim)
        )


class TransformerTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        num_heads=2,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        encoding_type="linear",
        max_seq_len=10,
        use_normalization=True,
    ):
        super(TransformerTimeSeriesClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.encoding_type = encoding_type
        self.max_seq_len = max_seq_len
        self.use_normalization = use_normalization

        if self.use_normalization:
            self.norm = nn.LayerNorm(self.input_size)

        self.positional_encoding = CustomPositionalEncoding(
            embed_dim=self.input_size,
            max_len=self.max_seq_len,
            encoding_type=self.encoding_type,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.input_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if self.use_normalization:
            x = self.norm(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.transpose(1, 2)
        x = self.pooling(x)
        x = self.classifier(x)
        # if using BCEWithLogitsLoss or CrossEntropyLoss, don't use sigmoid
        # only use sigmoid if using BCELoss
        # if using output layer with 1 neuron, use BCELoss
        # if using output layer with 2 neurons, use BCEWithLogitsLoss
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


model = TransformerTimeSeriesClassifier(
    input_size=100,
    num_classes=2,
    num_heads=2,
    num_layers=4,
    dim_feedforward=128,
    dropout=0.2,
    encoding_type="linear",
    max_seq_len=10,
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
