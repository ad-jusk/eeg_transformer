import torch
from torchmetrics.classification import Accuracy
from eeg_logger import logger

"""
From paper:

- Empirically, the number of head in each multi-head attention layer was set to 8
- The dropout rate was set to 0.3
- The parameter of the position-wise fully connected feed-forward layer with a ReLU activation was set to 512
- The weight attenuation was 0.0001
- All the models used the Adam optimizer. The training epoch was set to 50
- We set the number of training epochs to 10
- The EEG data were transformed into 3D tensors (N, C, T), where N is the number of trials, C is the number of channels, and T is the time points
- In our Transformer-based models, we set dk = dv = 64, which was the same size as EEG channel numbers.
"""

PREPROCESSED_DATA_DIR = "./preprocessed_data/Physionet"
D_MODEL = 64
NUM_HEADS = 8
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.0001
LEARNING_RATE = 0.0007


def train_model(
    model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device, verbose: bool
) -> None:
    """
    Trains model with parameters specified in paper.

    :param model: model to train
    :param train_loader: loader for training data
    :param device: device to train model on
    :param verbose: logs more info if set to true
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Computes accuracy of provided model.

    :param model: model to evaluate
    :param test_loader: loader for testing data
    :param device: device to evaluate model on
    """
    acc = Accuracy(task="binary").to(device)
    model.eval()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            preds = torch.argmax(output, dim=1)
            acc.update(preds, y_batch)

    return acc.compute().item()
