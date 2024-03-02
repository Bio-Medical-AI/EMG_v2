from typing import Any, Dict, Tuple, Union, Sequence, Optional
import os

import hydra
import torch
import lightning as pl
from torch.optim import Optimizer
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class MNISTLitModule(pl.LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        compile: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        """Initialize a `MNISTLitModule`.

        Args:
            compile (Optional[bool], optional). Whether to compile the model. Defaults to False.
            *args, **kwargs: Hyperaparameters for the model. Saved by `self.save_hyperparameters()`
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = hydra.utils.instantiate(self.hparams.model)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def configure_optimizers(
            self,
        ) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
            """Configure optimizer and lr scheduler.
            Returns:
                Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
                    Optimizer or optimizer and lr scheduler.
            """
            params = self.model.parameters()
            optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, params=params, _convert_="partial"
            )
            # return only optimizer if lr_scheduler is not provided.
            if "lr_scheduler" not in self.hparams:
                return {'optimizer': optimizer}
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
            )
            # reduce LR on Plateau requires special treatment
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor.metric,
                }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()


    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y


    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self._shared_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self._shared_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the training set.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): A batch of data (a tuple) containing the input tensor of images and target
                labels.
            batch_idx (int): The index of the current batch.
        """
        loss, preds, targets = self._shared_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage (str): Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)



@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _run(config: DictConfig) -> None:
    """
    Run to test if the module works.
    """
    model = hydra.utils.instantiate(config.lightning_model, _recursive_=False)
    print(f'{model}')


if __name__ == "__main__":
    _run()
