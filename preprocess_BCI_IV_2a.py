import mne
import numpy as np
import os

"""
CAN USE plot_BCI_2a to look at plotted data (only training)

only left and right hand, there is no rest

# Event |  Type  | Description
_________________________________________
# 276   | 0x0114 | Idling EEG (eyes open)
# 277   | 0x0115 | Idling EEG (eyes closed)
# 768   | 0x0300 | Start of a trial
# 769   | 0x0301 | Cue onset left (class 1)
# 770   | 0x0302 | Cue onset right (class 2)
# 771   | 0x0303 | Cue onset foot (class 3)
# 772   | 0x0304 | Cue onset tongue (class 4)
# 783   | 0x030F | Cue unknown
# 1023  | 0x03FF | Rejected trial
# 1072  | 0x0430 | Eye movements
# 32766 | 0x7FFE | Start of a new run

"""

def read_dataset_BCI_IV_2a(dirPath: str, saveDir: str):
    subjects_folders = [f for f in os.listdir(dirPath) if f.startswith("S")]

    bci_dir = os.path.join(saveDir, "BCI_IV_2a")
    os.makedirs(bci_dir, exist_ok=True)


    for subject in subjects_folders:
        train_data_file = os.path.join(dirPath, subject, f"A{subject[1:3]}T.gdf" )
        test_data_file = os.path.join(dirPath, subject, f"A{subject[1:3]}E.gdf" )

        # READS TRAIN DATA FROM PATH
        if os.path.exists(train_data_file):
            print(f"Reading training data from {subject}...")
            raw_train = mne.io.read_raw_gdf(train_data_file, preload=True)
        else:
            print(f"Missing training file: {train_data_file}")

        # READS TEST DATA FROM PATH
        if os.path.exists(test_data_file):
            print(f"Reading test data from {subject}...")
            raw_test = mne.io.read_raw_gdf(test_data_file, preload=True)
        else:
            print(f"Missing test file: {test_data_file}")


        train_data = preprocess_dataset_BCI_IV_2a(raw_train)
        test_data = preprocess_dataset_BCI_IV_2a(raw_test)

        # Save preprocessed data for the subject
        if train_data is not None:
            train_filename = os.path.join(bci_dir, f"PA{subject[1:3]}T.fif")

            if os.path.exists(train_filename):
                os.remove(train_filename)

            train_data.save(train_filename)
            print(f"Training data for subject {subject[1:3]} saved as {train_filename}")

        if test_data is not None:
            test_filename = os.path.join(bci_dir, f"PA{subject[1:3]}E.fif")

            if os.path.exists(test_filename):
                os.remove(test_filename)

            test_data.save(test_filename)
            print(f"Testing data for subject {subject[1:3]} saved as {test_filename}")
        


def preprocess_dataset_BCI_IV_2a(raw_data):
    events, event_id = mne.events_from_annotations(raw_data) # Extract events
    print("Extracted Events:", event_id)

    selected_event_id = {
        '769': 5,  # Left Hand
        '770': 6   # Right Hand
    }

    raw_data.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']

    # PICK ONLY EEG
    picks = mne.pick_types(raw_data.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

    tmin, tmax = 1., 4.
    epochs = mne.Epochs(raw_data, events, event_id=selected_event_id, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True)

    print("EEG_DATA" + f"{epochs}")
    return epochs 


def main() -> None:
    root_dir = ".\data\BCI_IV_2A"
    save_dir = "processed_data"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    read_dataset_BCI_IV_2a(root_dir, save_dir)

if __name__ == "__main__":
    main()