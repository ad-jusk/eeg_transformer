import requests, zipfile, os, shutil, sys
import numpy as np
from mne.datasets.eegbci import load_data
from eeg_logger import logger


def download_BCI_IV_2a(path: str) -> None:
    url: str = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip"
    zip_path: str = "./data/BCI_IV_2a.zip"

    if os.path.exists(path):
        logger.info(f"Dataset already downloaded in {path}")
        return

    logger.info(f"Downloading data from {url}...")
    r = requests.get(url, stream=True)

    if r.status_code != 200:
        logger.error(f"Failed to download BCI 2a dataset. Status code: {r.status_code}")
        return

    with open(zip_path, "wb") as file:
        for chunk in r.iter_content(1024):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)
    logger.info(f"Extraction complete: Files saved in {path}")

    os.remove(zip_path)
    create_subject_dir_structure_BCI_IV(path)


def download_BCI_IV_2b(path: str) -> None:
    url: str = "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip"
    zip_path: str = "./data/BCI_IV_2b.zip"

    if os.path.exists(path):
        logger.info(f"Dataset already downloaded in {path}")
        return

    logger.info(f"Downloading data from {url}...")
    r = requests.get(url, stream=True)

    if r.status_code != 200:
        logger.error(f"Failed to download BCI 2b dataset. Status code: {r.status_code}")
        return

    with open(zip_path, "wb") as file:
        for chunk in r.iter_content(1024):
            file.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(path)
    logger.info(f"Extraction complete: Files saved in {path}")

    os.remove(zip_path)
    create_subject_dir_structure_BCI_IV(path)


def create_subject_dir_structure_BCI_IV(path: str) -> None:

    for filename in os.listdir(path):
        subject_dir: str = f"{path}/S{filename[1:3]}"

        if not os.path.exists(subject_dir):
            os.makedirs(subject_dir)

        shutil.move(f"{path}/{filename}", subject_dir)

    logger.info(f"Subject file structure created in {path}")


def download_Physionet(path: str, num_patients: int = 109) -> None:

    if os.path.exists(path):
        logger.info(f"Dataset already downloaded in {path}")
        return

    runs = [4, 8, 12]  # RUNS FOR MOTOR IMAGERY
    subject_list: list[int] = np.arange(1, num_patients + 1, 1).tolist()  # PATIENTS TO DOWNLOAD
    custom_mne_dir = "MNE-eegbci-data"  # MNE IMPORTS DATA TO THIS DIRECTORY BY DEFAULT

    logger.info(f"Downloading Physionet data...")
    filenames: list[str] = load_data(subjects=subject_list, runs=runs, path=path, update_path=False)
    logger.info(f"Download complete: Files saved in {path}.")

    # HERE WE MOVE THE FILES FOR BETTER READABILITY
    for filename in filenames:
        parent_dir = os.path.basename(os.path.abspath(os.path.join(filename, os.pardir)))
        dest_path = f"{path}/{parent_dir}/"
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        shutil.move(filename, dest_path)
    logger.info(f"Files divided into subdirectories for all {num_patients} patients in {path}")

    # REMOVE CUSTOM MNE DIRECTORY
    shutil.rmtree(f"{path}/{custom_mne_dir}")


def main() -> None:
    dataset_name: str = sys.argv[1] if len(sys.argv) > 1 else ""
    data_base_dir: str = "./data"

    if not os.path.exists(data_base_dir):
        os.makedirs(data_base_dir)

    match dataset_name:
        case "bci2a":
            download_BCI_IV_2a(path=f"{data_base_dir}/BCI_IV_2a")
        case "bci2b":
            download_BCI_IV_2b(path=f"{data_base_dir}/BCI_IV_2b")
        case "physionet":
            download_Physionet(path=f"{data_base_dir}/Physionet", num_patients=2)
        case _:
            logger.info("Downloading all datasets")
            download_BCI_IV_2a(path=f"{data_base_dir}/BCI_IV_2a")
            download_BCI_IV_2b(path=f"{data_base_dir}/BCI_IV_2b")
            download_Physionet(path=f"{data_base_dir}/Physionet", num_patients=2)


if __name__ == "__main__":
    main()
