import argparse
import os
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import gdown
import hydra
import numpy as np
import pandas as pd
import scipy.io
from omegaconf import DictConfig
from tqdm import tqdm


def prepare_datasets(
    prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None], final_path: str
) -> None:
    """Download and prepare 3 base datamodule.

    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datamodule will be saved.
    """
    prepare_capgmyo(prepare_dataset, final_path)
    prepare_ninapro(prepare_dataset, final_path)
    prepare_myoarmband(prepare_dataset, final_path)


def prepare_capgmyo(
    prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None], final_path: str
) -> None:
    """Download and prepare CapgMyo dataset.

    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datamodule will be saved.
    """
    prepare_dataset(
        "CapgMyo", "1Xjtkr-rl2m3_80BvNg1wYSXN8yNaevZl", get_capgmyo_dataset, final_path
    )


def prepare_ninapro(
    prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None], final_path: str
) -> None:
    """Download and prepare NinaPro dataset.

    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datamodule will be saved.
    """
    prepare_dataset(
        "NinaPro_1", "1BtNxCiGIqVWYiPtf0AxyZuTY0uz8j6we", get_ninapro_dataset, final_path
    )


def prepare_myoarmband(
    prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None], final_path: str
) -> None:
    """Download and prepare MyoArmband dataset.

    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datamodule will be saved.
    """
    prepare_dataset(
        "MyoArmband", "1dO72tvtx5HOzZZ0C56IIUdCIVO3nNGt5", get_myoarmband_dataset, final_path
    )


def prepare_frame_dataset(
    dataset_name: str,
    file_id: str,
    data_loading_function: Callable[[], pd.DataFrame],
    final_folder: str,
) -> None:
    """Prepare dataset in form of separate files with arrays representing samples.

    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
        data_loading_function: Function that will return loaded dataset
        final_folder: folder in which datamodule will be saved.
    """
    prepare_folders(dataset_name, file_id)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    save_arrays(data_loading_function(), dataset_name, final_folder)


def prepare_dataframe_dataset(
    dataset_name: str,
    file_id: str,
    data_loading_function: Callable[[], pd.DataFrame],
    final_folder: str,
) -> None:
    """Prepare dataset in form of dataframe stored as .pkl file.

    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
        data_loading_function: Function that will return loaded dataset
        final_folder: folder in which datamodule will be saved.
    """
    prepare_folders(dataset_name, file_id)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    subfolder = os.path.join(final_folder, dataset_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    data_loading_function().to_pickle(
        os.path.join(final_folder, dataset_name, f"{dataset_name}.pkl")
    )


def prepare_folders(dataset_name: str, file_id: str) -> None:
    """Prepare folders for storing data, downloads and extracts data into them.

    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
    """
    if not os.path.exists(os.getenv("STORAGE_DIR")):
        os.makedirs(os.getenv("STORAGE_DIR"))
    destined_folder = os.path.join(os.getenv("STORAGE_DIR"), dataset_name)
    if not os.path.exists(destined_folder):
        os.makedirs(destined_folder)
    zip_file = os.path.join(destined_folder, dataset_name + ".zip")
    if not os.path.exists(zip_file):
        import_datasets(zip_file, file_id)
        with ZipFile(zip_file, "r") as zObject:
            zObject.extractall(destined_folder)


def import_datasets(destination: str, id: str) -> None:
    """Download compressed data files from Google Drive.

    Args:
        destination: Path of file to which download compressed dataset.
        id: Identifier of dataset on Google Drive for download with gdown.
    """
    gdown.download(id=id, output=destination, quiet=False)


def extract_data(relative_path: bytes or str, name: str) -> np.ndarray:
    """
    Extract data from MatLab file to an array.
    Args:
        relative_path: Relative path to .mat file
        name: Name of the column with desired data in .mat file

    Returns: Array with desired data
    """
    path = os.path.join(os.getenv("STORAGE_DIR"), relative_path)
    return np.array(scipy.io.loadmat(path)[name])


def get_absolute_path(relative_path: bytes or str) -> bytes or str:
    """
    Get Absolute path from relative_path
    Args:
        relative_path: Relative path (in relation to current file)

    Returns: Absolute path
    """
    return os.path.join(Path(os.path.abspath(__file__)).parent, relative_path)


def get_capgmyo_dataset() -> pd.DataFrame:
    """Go through original files of CapgMyo dataset and convert them into form useful for
    computation in python.

    Returns: Dataframe with dataset.
    """

    def int_in_3(num: int) -> str:
        if num < 0:
            return "000"
        if num < 10:
            return f"00{num}"
        if num < 100:
            return f"0{num}"
        if num < 1000:
            return str(num)
        return "BIG"

    recordings = []
    labels = []
    series = []
    subjects = []
    splits = []
    for test_object in range(1, 19):
        for gesture in range(1, 9):
            for recording in range(1, 11):
                data = extract_data(
                    os.path.join(
                        "CapgMyo",
                        int_in_3(test_object)
                        + "-"
                        + int_in_3(gesture)
                        + "-"
                        + int_in_3(recording),
                    ),
                    "data",
                )
                split = "test" if recording % 2 == 0 else "train"
                size = data.shape[0]
                labels.extend([gesture - 1 for _ in range(size)])
                data = np.split(data.reshape((size, 16, 8)).transpose(0, 2, 1), size)
                recordings.extend(data)
                series.extend(
                    [
                        (recording - 1) + (gesture - 1) * 10 + (test_object - 1) * 80
                        for _ in range(size)
                    ]
                )
                subjects.extend([test_object for _ in range(size)])
                splits.extend([split for _ in range(size)])
    return pd.DataFrame(
        {
            "record": [i[0] for i in recordings],
            "label": labels,
            "spectrogram": series,
            "subject": subjects,
            "split": splits,
        }
    )


def get_ninapro_dataset() -> pd.DataFrame:
    """Go through original files of NinaPro dataset and convert them into form useful for
    computation in python.

    Returns: Dataframe with dataset.
    """
    recordings = []
    labels = []
    series = []
    subjects = []
    splits = []
    last_series = -1
    for subject in range(1, 28):
        start_gest = 0
        for session in range(1, 4):
            data = extract_data(
                os.path.join("NinaPro_1", f"S{str(subject)}_A1_E{str(session)}.mat"), "emg"
            )
            gesture = extract_data(
                os.path.join("NinaPro_1", f"S{str(subject)}_A1_E{str(session)}.mat"), "stimulus"
            )
            split = "test" if session % 3 == 0 else "train"
            recordings.extend([d.reshape(10, -1) for d in data])
            proper_gesture = [(g if g == 0 else g + start_gest) for g in gesture[:, 0]]
            start_gest = max(proper_gesture)
            labels.extend(proper_gesture)
            counter = 1 + last_series
            series.append(counter)
            previous = gesture[0, 0]
            for gest in gesture[1:, 0]:
                if gest != previous:
                    counter += 1
                previous = gest
                series.append(counter)
            last_series = counter
            subjects.extend([subject for _ in range(gesture.shape[0])])
            splits.extend([split for _ in range(gesture.shape[0])])
    df = pd.DataFrame(
        {
            "record": recordings,
            "label": labels,
            "spectrogram": series,
            "subject": subjects,
            "split": splits,
        }
    )
    df = df.loc[df["label"] != 0]
    df["label"] = df["label"] - 1
    df.reset_index(inplace=True)
    return df


def get_myoarmband_dataset() -> pd.DataFrame:
    """Go through original files of MyoArmband dataset and convert them into form useful for
    computation in python.

    Returns: Dataframe with dataset.
    """

    def format_data_to_train(vector_to_format):
        emg_vector = []
        records = []
        for value in vector_to_format:
            emg_vector.append(value)
            if len(emg_vector) >= 8:
                records.append(np.array(emg_vector, dtype=np.float32).reshape((8, 1)))
                emg_vector = []
        return pd.DataFrame({"record": records})

    def classe_to_df(path: str, subfolder: str, start_val: int = 0):
        df = pd.DataFrame(columns=["record", "label", "spectrogram", "subject"])
        for i in range(28):
            os.path.join(path, f"classe_{i}.dat")
            arr = np.fromfile(os.path.join(path, subfolder, f"classe_{i}.dat"), dtype=np.int16)
            arr = np.array(arr, dtype=np.float32)
            formatted = format_data_to_train(arr)
            formatted["label"] = i % 7
            formatted["spectrogram"] = start_val + i
            df = pd.concat([df, formatted], ignore_index=True)
        return df

    def get_dataset(path: str, subjects: list, subfolder: str, series_val: int = 0.0):
        df = pd.DataFrame()
        for i, subject in enumerate(subjects):
            tmp_df = classe_to_df(os.path.join(path, subject), subfolder, series_val + i * 28)
            tmp_df["subject"] = i + series_val // 28
            df = pd.concat([df, tmp_df])
        if "train" in subfolder.lower():
            df["split"] = "train"
        elif "test" in subfolder.lower():
            df["split"] = "test"
        return df

    path = "MyoArmband"
    eval_path = os.path.join(path, "EvaluationDataset")
    pre_path = os.path.join(path, "PreTrainingDataset")

    subjects = [
        "Female0",
        "Female1",
        "Male0",
        "Male1",
        "Male2",
        "Male3",
        "Male4",
        "Male5",
        "Male6",
        "Male7",
        "Male8",
        "Male9",
        "Male10",
        "Male11",
        "Male12",
        "Male13",
        "Male14",
        "Male15",
    ]
    dataset = get_dataset(eval_path, subjects, "training0")
    dataset = pd.concat(
        [dataset, get_dataset(eval_path, subjects, "Test0", 504)], ignore_index=True
    )
    dataset = pd.concat(
        [dataset, get_dataset(eval_path, subjects, "Test1", 1008)], ignore_index=True
    )
    subjects2 = [
        "Female0",
        "Female1",
        "Female2",
        "Female3",
        "Female4",
        "Female5",
        "Female6",
        "Female7",
        "Female8",
        "Female9",
        "Male0",
        "Male1",
        "Male2",
        "Male3",
        "Male4",
        "Male5",
        "Male6",
        "Male7",
        "Male8",
        "Male9",
        "Male10",
        "Male11",
    ]
    dataset2 = get_dataset(pre_path, subjects2, "training0", 1512)
    final_dataset = pd.concat([dataset, dataset2], ignore_index=True)
    return final_dataset


def save_arrays(dataframe: pd.DataFrame, name: str, path: os.path) -> pd.DataFrame:
    """
    Save all samples from dataset as ndarrays
    Args:
        dataframe: dataframe with samples to convert and save
        name: Name of dataset
        path: path to folder in which sub-folder for dataset will be created

    Returns:

    """
    paths = []
    base_path = os.path.join(path, name)
    for index, row in tqdm(dataframe.iterrows(), total=len(dataframe.index)):
        current_path = os.path.join(base_path, f"{index // 200000}")
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        path = os.path.join(current_path, f"{name}_{index}.npy")
        np.save(path, row["record"])
        paths.append(path)

    df: pd.DataFrame = pd.DataFrame(
        {
            "path": paths,
            "label": dataframe["label"],
            "spectrogram": dataframe["spectrogram"],
            "subject": dataframe["subject"],
            "split": dataframe["split"],
        }
    )
    df.to_csv(os.path.join(base_path, f"{name}.csv"), index=False)
    return df


DOWNLOADS = {
    "NinaPro_1": get_ninapro_dataset,
    "CapgMyo": get_capgmyo_dataset,
    "MyoArmband": get_myoarmband_dataset,
}


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def main(config: DictConfig):
    """Main entry point for downloading datasets.

    Args:
        config: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float]: Optimized metric value.
    """
    prepare_frame_dataset(
        config.name, config.gdrive_id, DOWNLOADS[config.name], os.getenv("DATA_DIR")
    )


if __name__ == "__main__":
    main()
