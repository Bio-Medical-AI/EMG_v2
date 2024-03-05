import numpy as np
import pandas as pd
import torch
from classification.utils.measurements import calculate_mean_std
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
from utils.rich_utils import progress_load


class AbstractDataset(Dataset):
    """Base dataset for unordered images."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        transform: Compose = ToTensor(),
        source_name: str = "path",
        target_name: str = "label",
        series_name: str = "spectrogram",
        preload: bool = True,
        progress: CustomProgress = None,
    ):
        """
        Args:
            data_frame: DataFrame containing all data included in dataset
            transform: Transforms that are meant to be applied to single data sample
            source_name: Name of column in dataframe with data samples
            target_name: Name of column in dataframe with target classes
            series_name: Name of column in dataframe with series ID
        """
        self.preload = preload
        self.records: pd.Series = data_frame[source_name]
        self.progress: CustomProgress = progress

        if self.preload:
            if self.progress is not None:
                task = self.progress.add_task("[cyan]Preloading", total=self.records.size)
                self.records = self.records.apply(lambda x: progress_load(x, task, self.progress))
            else:
                self.records = self.records.apply(np.load)
        self.labels: pd.Series = data_frame[target_name]
        self.series: pd.Series = data_frame[series_name]
        self.transform = transform
        self.samples_amount = len(data_frame.index)

    def get_mean_std(self) -> tuple[float, float]:
        """Get mean and std of this dataset.

        Returns:
            Mean and Std
        """
        if not self.preload:
            mean, std = calculate_mean_std(self.records.tolist(), self.progress)
        else:
            values = np.stack([item for _, item in self.records.items()])
            mean, std = np.mean(values), np.std(values)
        return mean, std

    def set_transform_mean_std(self, mean: float, std: float):
        """Set mean and std for Normalizing transform of this dataset.

        Args:
            mean: mean
            std: standard deviation
        """
        norm_idx = None
        for idx, tr in enumerate(self.transform.transforms):
            if type(tr) is Normalize:
                norm_idx = idx
        if norm_idx is not None:
            self.transform.transforms[norm_idx].mean = mean
            self.transform.transforms[norm_idx].std = std

    def __getitem__(self, index: int) -> dict:
        label = torch.tensor(self.labels.iloc[index]).long()
        series = torch.tensor(self.series.iloc[index]).long()
        out_index = torch.tensor(index).long()
        if self.preload:
            data = self.records.iloc[index]
        else:
            data = np.load(self.records.iloc[index])
        if self.transform is not None:
            data = self.transform(data).float()
        return {"data": data, "label": label, "spectrograms": series, "index": out_index}

    def __len__(self) -> int:
        return self.samples_amount
