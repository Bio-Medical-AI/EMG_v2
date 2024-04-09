from typing import List

import numpy as np
import pandas as pd
from classification.dataset import AbstractDataset
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from torchvision.transforms import Compose, ToTensor


class SpectrogramDataset(AbstractDataset):
    """Base dataset for sequences of images."""

    def __init__(
        self,
        data_frame: pd.DataFrame,
        transform: Compose = ToTensor(),
        source_name: str = "path",
        target_name: str = "label",
        series_name: str = "spectrogram",
        preload: bool = True,
        progress: CustomProgress = None,
        visualize_progress: bool = False,
        window_length: int = 1,
    ):
        """
        Args:
            data_frame: DataFrame containing all data included in dataset
            transform: Transforms that are meant to be applied to single data sample
            source_name: Name of column in dataframe with data samples
            target_name: Name of column in dataframe with target classes
            series_name: Name of column in dataframe with series ID
            preload: Should dataset be preloaded into ram memory?
            progress: Object from trainer responsible for displaying progressbar
            visualize_progress: Should preloading progress be visualized?
            window_length: length of series of data
        """
        super().__init__(
            data_frame,
            transform,
            source_name,
            target_name,
            series_name,
            preload,
            progress,
            visualize_progress,
        )
        self.window_length = window_length

    def __getitem__(self, index: int) -> dict:
        """Get one item from dataset under given index.

        Args:
            index: ndex of element in dataset
        """
        label = int(self.labels.iloc[index])
        series = self.series.iloc[index]
        data = np.squeeze(np.dstack(self._get_window(index, self.window_length)))
        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=1)
        if self.transform is not None:
            data = self.transform(data).float()
        return {"data": data, "label": label, "spectrograms": series, "index": index}

    def __len__(self) -> int:
        """Get length of dataset."""
        return self.samples_amount

    def _get_window(self, index: int, length: int) -> list[np.ndarray]:
        """Concat multiple frames of recording into one tensor.

        Args:
            index: index of frame to which all the previous frames will be appended.
            length: number of frames to concat
        """
        if self.series.iloc[index] == self.series.iloc[index - length + 1]:
            data = self.records.iloc[index - length + 1 : index + 1].tolist()
            if not self.preload:
                data = [
                    np.load(item)
                    for item in self.records.iloc[index - length + 1 : index + 1].tolist()
                ]
            return data

        if self.preload:
            data = self.records.iloc[index]
        else:
            data = np.load(self.records.iloc[index])
        if index - length + 1 < 0:
            return self._get_window(index, length - 1) + [data]
        return self._get_window(index, length - 1) + [data]
