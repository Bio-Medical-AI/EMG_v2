import statistics
from typing import Any

import hydra
import lightning as L
import pandas as pd
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from utils.rich_utils import get_progress_callback


class Classifier(L.LightningModule):
    """Module performing all tasks of classification of given data with some model.

    It is responsible for training, validation, testing and prediction. It defines how those
    processes are organised. It measures all metrics and performs majority voting.
    """

    def __init__(self, **kwargs):
        """
        Args:
            model: Model to be used for classification
            optimizer: Optimizer for model training
            lr_scheduler: Learning rate scheduler, which will be used in training
            criterion: Criterion function for training
            time_window: List of Numbers of samples to perform majority voting of which
            time_step: List of numbers of samples that are ignored until next majority voting is performed. Their order corresponds to order in time_window
            window_fix: List of numbers that are equal to number of records in one sample minus one
            metrics: Collection of Metrics to be computed for classification results
            monitor: Name of metric to monitor for scheduler. Some schedulers are changing learning rate based on that metric.
            **kwargs
        """
        super().__init__()
        self.save_hyperparameters()
        self.epoch_results = []
        self.model = hydra.utils.instantiate(self.hparams.model)
        self.criterion = hydra.utils.instantiate(self.hparams.criterion)
        self.metrics = hydra.utils.instantiate(self.hparams.metrics)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computing prediction for given matrix.
        Args:
            x: Tensor representing picture

        Returns:
            Vector of values representing probability of picture being each class
        """
        return self.model(x)

    def configure_optimizers(self) -> dict[str, dict[str, LambdaLR or None] or Any]:
        """Create optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and configured learning rate scheduler
        """

        params = self.model.parameters()
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=params, _convert_="partial"
        )
        if "lr_scheduler" not in self.hparams:
            return {"optimizer": optimizer}
        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": self.hparams.monitor,
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch: dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        """
            Perform a training step and log loss
        Args:
            train_batch: Batch of training data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(train_batch)
        logs = {"loss": results["loss"]}
        logs = self._add_prefix_to_metrics("train/", logs)
        self.log_dict(logs)
        self.epoch_results.append(results)
        return results

    def on_training_epoch_end(self) -> None:
        """Finish training epoch and log all metrics results."""
        results = self._epoch_end(self.epoch_results)
        logs = self._add_prefix_to_metrics("train/", results["measurements"])
        self.log_dict(logs)
        self.epoch_results = []

    def validation_step(self, val_batch: dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        """
            Perform a validation step
        Args:
            val_batch: Batch of validation data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(val_batch)
        self.epoch_results.append(results)
        return results

    def on_validation_epoch_end(self) -> None:
        """Finish validation epoch and log all metrics results and loss."""
        results = self._epoch_end(self.epoch_results)
        results = self._eval_epoch_end(results)["measurements"]
        logs = self._add_prefix_to_metrics("val/", results)
        self.log_dict(logs)
        self.epoch_results = []

    def test_step(self, test_batch: dict[str, Tensor or Any], batch_idx: int) -> STEP_OUTPUT:
        """
            Perform a test step
        Args:
            test_batch: Batch of test data
            batch_idx: index of batch

        Returns:
            Computed predictions
        """
        results = self._step(test_batch)
        self.epoch_results.append(results)
        return results

    def on_test_epoch_end(self) -> None:
        """Finish test epoch and log all metrics results and loss."""
        results = self._epoch_end(self.epoch_results)
        results = self._eval_epoch_end(results)
        results = self._vote(results)["measurements"]
        logs = self._add_prefix_to_metrics("test/", results)

        self.log_dict(logs)
        self.epoch_results = []
        self.trainer.logger.finalize("success")

    @staticmethod
    def _add_prefix_to_metrics(prefix: str, logs: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Add prefix to all keys in dictionary
        Args:
            prefix: Train, val or test
            logs: Dictionary with measured metrics

        Returns:
            Dictionary with added prefixes
        """
        logs = {(prefix + key): value for key, value in logs.items()}
        return logs

    def _calculate_metrics(self, preds: Tensor, targets: Tensor) -> dict[str, Tensor or list]:
        """
        Calculate all metrics
        Args:
            preds: Predictions.
            targets: Labels.

        Returns:
            Calculated metrics
        """
        metrics = self.metrics(preds, targets)
        return metrics

    def predict_step(self, predict_batch: Tensor, batch_idx: int, dataloader_idx=0) -> STEP_OUTPUT:
        """
            Perform pure prediction on data
        Args:
            predict_batch: batch of data
            batch_idx: index of batch
            dataloader_idx: index of dataloader

        Returns:
            Predicted classes
        """
        return self.model(predict_batch)

    def _step(self, batch: dict[str, Tensor or Any]) -> dict[str, Tensor or Any]:
        """
            Base of training, validation and test steps
        Args:
            batch: Batch of data

        Returns:
            Dictionary with: loss, predictions, labels, series numbers and indexes
        """
        x = batch["data"]
        y = batch["label"]
        series = batch["spectrograms"]
        index = batch["index"]
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {"loss": loss, "preds": preds, "labels": y, "spectrograms": series, "index": index}

    @staticmethod
    def _moving_average(df: pd.DataFrame, window: int, step: int) -> STEP_OUTPUT:
        """
            Compute a moving average over dataframe
        Args:
            df: Dataframe with 2 columns: preds and labels
            window: Amount of values to compute moving average on.
            step: Step of moving average.

        Returns:
            Computed moving average of 2 columns: preds and labels
        """
        if df.shape[0] >= window:
            preds = []
            labels = []
            for i in range((df.shape[0] - window) // step + 1):
                tmp = df.iloc[(i * step) : (i * step + window)]
                preds.append(tmp["preds"].values.tolist())
                labels.append(tmp["labels"].values.tolist())
            return {"preds_list": preds, "labels_list": labels}
        else:
            return {"preds": df["preds"].mode()[0].item(), "labels": df["labels"].mode()[0].item()}

    def _majority_voting(self, df: pd.DataFrame, window: int, step: int) -> dict[str, Any]:
        """
            Perform a majority voting over dataframe and compute metrics for the results.
        Args:
            df: Dataframe with 3 columns: preds, labels and spectrograms
            window: Amount of values to compute moving average on.
            step: Step of moving average.

        Returns:
            Dictionary with computed metrics for voted majorities
        """
        preds = []
        labels = []
        preds_list = []
        labels_list = []

        spcs = df["spectrograms"].unique().tolist()

        progress = get_progress_callback(self.trainer)
        task = progress.add_task(f"[cyan]Majority Voting {window}/{step}", total=len(spcs))

        for series in spcs:
            results = self._moving_average(df.loc[df["spectrograms"] == series], window, step)
            if "preds_list" in results.keys():
                preds_list += results["preds_list"]
                labels_list += results["labels_list"]
            else:
                preds.append(results["preds"])
                labels.append(results["labels"])
            progress.update(task, advance=1)
        preds += torch.mode(torch.Tensor(preds_list))[0].int().tolist()
        labels += torch.mode(torch.Tensor(labels_list))[0].int().tolist()

        majority_preds = torch.tensor(preds, device=self.device)
        majority_labels = torch.tensor(labels, device=self.device)

        return self._add_prefix_to_metrics(
            "majority_voting_", self._calculate_metrics(majority_preds, majority_labels)
        )

    def _epoch_end(
        self, step_outputs: list[STEP_OUTPUT]
    ) -> dict[str, dict[str, STEP_OUTPUT | float]]:
        """
            End an epoch and log all the computed metrics.
        Args:
            step_outputs: Collected results of all steps

        Returns:
            Dictionary containing computed measurements and outputs from model
        """
        preds = self._connect_epoch_results(step_outputs, "preds")
        labels = self._connect_epoch_results(step_outputs, "labels")
        series = self._connect_epoch_results(step_outputs, "spectrograms")
        index = self._connect_epoch_results(step_outputs, "index")
        loss = [step["loss"] for step in step_outputs]
        output = {
            "preds": preds,
            "labels": labels,
            "spectrograms": series,
            "index": index,
            "loss": loss,
        }
        measurements = self._calculate_metrics(preds.to(self.device), labels.to(self.device))
        return {"output": output, "measurements": measurements}

    @staticmethod
    def _eval_epoch_end(step_outputs: dict[str, dict[str, STEP_OUTPUT | float]]) -> STEP_OUTPUT:
        """
            End validation or test epoch and log all the computed metrics.
        Args:
            step_outputs: Collected results of all steps

        Returns:
            Dictionary containing computed measurements and outputs from model
        """
        output = step_outputs["output"]
        measurements = step_outputs["measurements"]
        measurements.update({"loss": statistics.fmean(output["loss"])})
        output.pop("loss", None)
        return {"output": output, "measurements": measurements}

    def _vote(self, step_outputs: STEP_OUTPUT) -> STEP_OUTPUT:
        """
            Start series of majority voting for each size of voting window defined in classifier.
        Args:
            step_outputs:

        Returns: Dictionary containing computed measurements and outputs from model

        """
        output = step_outputs["output"]
        measurements = step_outputs["measurements"]
        df = pd.DataFrame(output).sort_values(by=["index"])
        if self.hparams.window_fix is None:
            for window, step in zip(self.hparams.time_window, self.hparams.time_step):
                results = self._majority_voting(df, window, step)
                measurements.update(self._add_prefix_to_metrics(f"{window}_{step}/", results))
        else:
            for window, step, fix in zip(
                self.hparams.time_window, self.hparams.time_step, self.hparams.window_fix
            ):
                results = self._majority_voting(df, window, step)
                measurements.update(
                    self._add_prefix_to_metrics(f"{window + fix}_{step}/", results)
                )
        return {"output": output, "measurements": measurements}

    @staticmethod
    def _connect_epoch_results(step_outputs: list[STEP_OUTPUT], key: str) -> Tensor:
        """
            Connect values from dictionaries from all steps into one for the whole epoch
        Args:
            step_outputs: List of dictionaries to connect
            key: key to dictionary to specify, which values will be connected

        Returns:
            Connected values
        """
        to_concat = []
        for output in step_outputs:
            to_concat.append(output[key].detach().cpu())
        return torch.cat(to_concat)
