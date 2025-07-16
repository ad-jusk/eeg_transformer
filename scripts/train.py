import numpy as np
import torch
import mne
import os
import sys
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from scripts.dataset.eeg_dataset import EEGDataset
from scripts.models.transformer_models import (
    SpatialTransformer,
    TemporalTransformer,
    SpatialCNNTransformer,
    TemporalCNNTransformer,
    FusionCNNTransformer,
)
import scripts.models.utils as utils
from eeg_logger import logger


def load_subject_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    epochs = mne.read_epochs(file_path, preload=True, verbose=False)
    """
    Format danych: (Number of epochs, channels, n_times)
    Dla danych 3-sekundowych: 3 sekundy x 160 Hz = 480, n_times = 480
    Dla danych 6-sekundowych: 6 sekund x 160 Hz = 960, n_times = 960
    """
    # Data
    X = epochs.get_data()
    # Labels
    y = epochs.events[:, -1]
    # Labels should be numered 0, 1, 2 ...
    y = np.array([0 if label == 2 else 1 for label in y])
    return X, y


def create_model(model_name: str, test_data_shape: np.ndarray.shape) -> torch.nn.Module:
    match model_name:
        case "SpatialTransformer":
            return SpatialTransformer(
                input_size=test_data_shape[2],
                d_model=utils.D_MODEL,
                num_heads=utils.NUM_HEADS,
                num_classes=utils.NUM_CLASSES,
            )
        case "TemporalTransformer":
            return TemporalTransformer(
                input_size=test_data_shape[1],
                d_model=utils.D_MODEL,
                num_heads=utils.NUM_HEADS,
                num_classes=utils.NUM_CLASSES,
            )
        case "SpatialCNNTransformer":
            return SpatialCNNTransformer(
                d_model=utils.D_MODEL,
                num_heads=utils.NUM_HEADS,
                num_classes=utils.NUM_CLASSES,
            )
        case "TemporalCNNTransformer":
            return TemporalCNNTransformer(
                d_model=utils.D_MODEL,
                num_heads=utils.NUM_HEADS,
                num_classes=utils.NUM_CLASSES,
            )
        case "FusionCNNTransformer":
            return FusionCNNTransformer(
                d_model=utils.D_MODEL,
                num_heads=utils.NUM_HEADS,
                num_classes=utils.NUM_CLASSES,
            )


def train_model(model_name: str, cnn_mode: bool = False) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        logger.warning("Warning - training model on cpu")

    accuracies = []
    all_subject_folders = sorted(os.listdir(utils.PREPROCESSED_DATA_DIR))
    all_X = []
    all_y = []

    for subj_folder in all_subject_folders:
        subj_folder_path = os.path.join(utils.PREPROCESSED_DATA_DIR, subj_folder)
        file_path = os.path.join(subj_folder_path, f"PA{subj_folder[1:]}-3s-epo.fif")
        if os.path.exists(file_path):
            X, y = load_subject_data(file_path)
            all_X.append(X)
            all_y.append(y)

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_X, all_y)):

        X_train, X_test = all_X[train_idx], all_X[test_idx]
        y_train, y_test = all_y[train_idx], all_y[test_idx]

        train_dataset = EEGDataset(X_train, y_train, cnn_mode=cnn_mode)
        test_dataset = EEGDataset(X_test, y_test, cnn_mode=cnn_mode)

        train_loader = DataLoader(train_dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=utils.BATCH_SIZE, shuffle=False)

        model = create_model(model_name, X_train.shape)

        logger.info(f"Training {model_name} in fold {fold + 1}...")
        utils.train_model(model, train_loader, device, verbose=False)

        accuracy = utils.evaluate_model(model, test_loader, device)
        logger.info(f"Accuracy for {model_name}  in fold {fold + 1}: {accuracy * 100:.2f}%")
        accuracies.append(accuracy)

    logger.info(f"Accuracy across 5 folds for {model_name}: {np.mean(accuracies) * 100:.2f}%")


def main() -> None:

    model_to_train: str = sys.argv[1] if len(sys.argv) > 1 else ""

    match model_to_train:
        case "spatial":
            train_model("SpatialTransformer")
        case "temporal":
            train_model("TemporalTransformer")
        case "spatialcnn":
            train_model("SpatialCNNTransformer", cnn_mode=True)
        case "temporalcnn":
            train_model("TemporalCNNTransformer", cnn_mode=True)
        case "fusion":
            train_model("FusionCNNTransformer", cnn_mode=True)


if __name__ == "__main__":
    main()
