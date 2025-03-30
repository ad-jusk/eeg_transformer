import mne
import os
import matplotlib.pyplot as plt
import math

def plot_evoked_responses(save_dir: str):
    # List all the subject files in the processed_data folder
    subject_files = [f for f in os.listdir(save_dir) if f.endswith('T.fif')]
    print(f"Files found: {subject_files}")  

    num_subjects = len(subject_files)
    num_cols = 2  
    num_rows = num_subjects  

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))

    for idx, subject_file in enumerate(subject_files):
        file_path = os.path.join(save_dir, subject_file)
        epochs = mne.read_epochs(file_path)
        print(f"subject_file ({subject_file}):")
        
        print(f"Events in {subject_file}: {epochs.event_id}")

        # Check if epochs are loaded correctly
        if len(epochs) == 0:
            print(f"No epochs found for {subject_file}")
            continue  

        # Plot for Left Hand (Event ID 769)
        if '769' in epochs.event_id:
            evoked_left = epochs['769'].average()
            print(f"Evoked response for Left Hand ({subject_file}):")
            print(evoked_left)
            ax = axs[idx, 0] 
            evoked_left.plot(time_unit='s', axes=ax, show=False)
            ax.set_title(f"Subject {subject_file} - Left Hand")

        # Plot for Right Hand (Event ID 770)
        if '770' in epochs.event_id:
            evoked_right = epochs['770'].average()
            print(f"Evoked response for Right Hand ({subject_file}):")
            print(evoked_right)
            ax = axs[idx, 1] 
            evoked_right.plot(time_unit='s', axes=ax, show=False)
            ax.set_title(f"Subject {subject_file} - Right Hand")

    plt.tight_layout()

def main():
    processed_data_dir = "processed_data\BCI_IV_2a"  

    plot_evoked_responses(processed_data_dir)
    plt.show()

if __name__ == "__main__":
    main()
