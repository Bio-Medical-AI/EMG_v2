import numpy as np
import pandas as pd
import torch
from classification.dataset import SpectrogramDataset
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from torchvision.transforms import Compose, ToTensor


class BiometricDataset(SpectrogramDataset):
    """Base dataset for Biometry experiment."""

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
        mean: int = 0,
        std: int = 0,
        n_th: int = 1,
        scale: int = 1,
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
            mean: mean of noise to add
            std: mean of noise to add
            n_th: every nth sample of signal will be preserved while others will be ignored
            scale: factor of signal rescaling
        """
        self.n_th = n_th
        if self.n_th > 1:
            data_frame = data_frame.iloc[:: self.n_th, :].reset_index(drop=True)
        super().__init__(
            data_frame,
            transform,
            source_name,
            target_name,
            series_name,
            preload,
            progress,
            visualize_progress,
            window_length,
        )
        self.mean = mean
        self.std = std
        self.scale = scale

    def __getitem__(self, index: int) -> dict:
        """Get one item from dataset under given index.

        Args:
            index: ndex of element in dataset
        """
        label = int(self.labels.iloc[index])
        series = self.series.iloc[index]
        data = np.squeeze(np.dstack(self._get_window(index, self.window_length)))
        data *= self.scale
        data += torch.normal(
            mean=torch.ones(size=data.shape) * self.mean,
            std=torch.ones(size=data.shape) * self.std,
        ).numpy()

        if len(data.shape) < 2:
            data = np.expand_dims(data, axis=1)
        if self.transform is not None:
            data = self.transform(data).float()
        return {"data": data, "label": label, "spectrograms": series, "index": index}
