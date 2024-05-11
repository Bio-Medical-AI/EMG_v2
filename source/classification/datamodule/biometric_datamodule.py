import hydra
from classification.datamodule import AbstractDataModule
from torch.utils.data import DataLoader
from utils.rich_utils import get_progress_callback


class BiometricDataModule(AbstractDataModule):
    """Datamodule for biometry experiments."""

    def train_dataloader(self) -> DataLoader:
        """Get Training dataloader."""
        if "train" not in self.datasets.keys():
            progress = get_progress_callback(self.trainer)
            self.datasets["train"] = hydra.utils.call(
                self.hparams.datasets.train,
                data_frame=self.data.iloc[self.splits["train"]].reset_index(),
                source_name=self.hparams.source_name,
                target_name=self.hparams.subject_name,
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
        """Get Validation dataloader."""
        if "val" not in self.datasets.keys():
            progress = get_progress_callback(self.trainer)
            self.datasets["val"] = hydra.utils.call(
                self.hparams.datasets.val,
                data_frame=self.data.iloc[self.splits["val"]].reset_index(),
                source_name=self.hparams.source_name,
                target_name=self.hparams.subject_name,
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
        """Get Test dataloader."""
        progress = get_progress_callback(self.trainer)
        dataset = hydra.utils.call(
            self.hparams.datasets.test,
            data_frame=self.data.iloc[self.splits["test"]].reset_index(),
            source_name=self.hparams.source_name,
            target_name=self.hparams.subject_name,
            series_name=self.hparams.series_name,
            window_length=self.hparams.window_length,
            progress=progress,
            _recursive_=True,
        )
        dataset.set_transform_mean_std(self.mean, self.std)
        return DataLoader(
            dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers
        )
