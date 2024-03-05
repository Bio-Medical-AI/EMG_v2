import os
import random
from random import randint
from typing import Dict, Optional

import hydra
import lightning as L
import pandas as pd
from lightning.pytorch.trainer.states import TrainerFn
from pandas import Index
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from torch.utils.data import DataLoader
from utils.rich_utils import get_progress_callback


class AbstractDataModule(L.LightningDataModule):
    """Basic DataModule to load data during experiments, designed for convolutional networks."""

    def __init__(self, **kwargs):
        """
        Args:
            name: Name of dataset, director for it will be created using that name.
            width: Width of sample from dataset
            height: Height of sample from dataset
            channels: Channels of sample from dataset
            num_classes: Number of classes that sample can be classified as such.
            window_length: Amount of samples from original dataset to compress into one.
            train_vs_val_size: Part of the whole data that should be used for training. Not important for cross-validation.
            k_folds: Number of folds to which data should be split. If less than 3 then data is split once based on train_vs_rest_size and val_vs_test_size.
            split_method: Method for splitting data. Can be set as 'equal', 'trials', 'subjects' or None.
                          If None, then data series are split randomly.
                          If 'equal', then each split will have the same subjects and gestures, in equal amount, but different trials in them.
                          If 'trials', then each split will have the same subjects but different trials in them.
                          If 'subjects', then each split will have different subjects in it.
            source_name: Name of column in dataframe with data
            target_name: Name of column in dataframe with labels
            series_name: Name of column in dataframe with series id
            subject_name: Name of column in dataframe with subject id
            split_name: Name of column in dataframe with split id
            batch_size: Size of single batch of data.
            num_workers: Number of workers used for loading data
            shuffle_train: Should training data be shuffled for every epoch
            seed: Seed value for all random generators taking part in process of loading data
        """
        super().__init__()
        self.save_hyperparameters()
        self.mean: float | None = 0
        self.std: float | None = 1
        if self.hparams.seed is not None:
            random.seed(self.hparams.seed)
        self.df_path = os.path.join(
            os.getenv("DATA_DIR"), self.hparams.name, f"{self.hparams.name}.csv"
        )
        self.data: pd.DataFrame = pd.DataFrame()
        self.splits: Dict[str, Index] = {}
        self.skf = None
        self.datasets = {}

    def prepare_data(self) -> None:
        self.data = pd.read_csv(self.df_path)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == TrainerFn.TESTING:
            self.datasets = {}
        self.split_data(stage)

    def split_data(self, stage: Optional[str]) -> None:
        """Split data into proper train, validate and test splits. Contents of Test set are decided
        by split_name column, but Validation and Training are set like that only if k_folds value
        is less than 2. Else we will perform cross-validation in which splitting strategy will be
        decided by split_method parameter of datamodule.

        - If 'equal', then each split will have the same subjects and gestures, in equal amount, but different trials in them.
        - If 'trial', then each split will have the same subjects but different trials in them.
        - If 'subject', then each split will have different subjects in it.
        - Else data series are split randomly

        Returns:
        """
        if "test" not in self.splits.keys():
            self.splits["test"] = self.data.loc[self.data[self.hparams.split_name] == "test"].index
            self.splits["rest"] = self.data.loc[self.data[self.hparams.split_name] != "test"].index
            if self.hparams.k_folds > 1:
                if self.hparams.split_method == "subject":
                    self.skf = GroupKFold(n_splits=self.hparams.k_folds).split(
                        self.splits["rest"],
                        self.data.iloc[self.splits["rest"]][self.hparams.target_name],
                        self.data.iloc[self.splits["rest"]][self.hparams.subject_name],
                    )
                elif self.hparams.split_method == "trial":
                    self.skf = StratifiedGroupKFold(n_splits=self.hparams.k_folds).split(
                        self.splits["rest"],
                        self.data.iloc[self.splits["rest"]][self.hparams.subject_name],
                        self.data.iloc[self.splits["rest"]][self.hparams.series_name],
                    )
                elif self.hparams.split_method == "equal":
                    self.skf = StratifiedKFold(n_splits=self.hparams.k_folds).split(
                        self.splits["rest"],
                        (
                            self.data.iloc[self.splits["rest"]][self.hparams.subject_name]
                            * self.data.iloc[self.splits["rest"]][self.hparams.target_name].max()
                            + self.data.iloc[self.splits["rest"]][self.hparams.target_name]
                        ),
                    )
                else:
                    self.skf = KFold(n_splits=self.hparams.k_folds).split(self.splits["rest"])
            else:
                val_tag = "val" if "val" in self.data[self.hparams.split_name].unique() else "test"
                self.splits["val"] = self.data.loc[
                    self.data[self.hparams.split_name] == val_tag
                ].index
                self.splits["train"] = self.data.loc[
                    self.data[self.hparams.split_name] == "train"
                ].index

        if self.hparams.k_folds > 1 and stage == TrainerFn.FITTING:
            train_index, test_index = next(self.skf)
            self.splits["train"] = self.splits["rest"][train_index]
            self.splits["val"] = self.splits["rest"][test_index]

    def train_dataloader(self) -> DataLoader:
        if "train" not in self.datasets.keys():
            progress = get_progress_callback(self.trainer)
            self.datasets["train"] = hydra.utils.call(
                self.hparams.datasets.train,
                data_frame=self.data.iloc[self.splits["train"]].reset_index(),
                source_name=self.hparams.source_name,
                target_name=self.hparams.target_name,
                series_name=self.hparams.series_name,
                window_length=self.hparams.window_length,
                progress=progress,
                _recursive_=True,
            )
            self.mean, self.std = self.datasets["train"].get_mean_std()
        self.datasets["train"].set_transform_mean_std(self.mean, self.std)
        return DataLoader(
            self.datasets["train"],
            shuffle=self.hparams.shuffle_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if "val" not in self.datasets.keys():
            progress = get_progress_callback(self.trainer)
            self.datasets["val"] = hydra.utils.call(
                self.hparams.datasets.val,
                data_frame=self.data.iloc[self.splits["val"]].reset_index(),
                source_name=self.hparams.source_name,
                target_name=self.hparams.target_name,
                series_name=self.hparams.series_name,
                window_length=self.hparams.window_length,
                progress=progress,
                _recursive_=True,
            )
        self.datasets["val"].set_transform_mean_std(self.mean, self.std)
        return DataLoader(
            self.datasets["val"],
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        progress = get_progress_callback(self.trainer)
        dataset = hydra.utils.call(
            self.hparams.datasets.test,
            data_frame=self.data.iloc[self.splits["test"]].reset_index(),
            source_name=self.hparams.source_name,
            target_name=self.hparams.target_name,
            series_name=self.hparams.series_name,
            window_length=self.hparams.window_length,
            progress=progress,
            _recursive_=True,
        )
        dataset.set_transform_mean_std(self.mean, self.std)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )
