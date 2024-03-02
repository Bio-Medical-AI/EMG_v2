from typing import Any, Dict, List, Optional, Tuple
import os

import hydra
import lightning as pl

import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from utils import RankedLogger, extras, get_metric_value, instantiate_callbacks, instantiate_loggers, log_hyperparameters, task_wrapper


logger = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        config (DictConfig): A DictConfig configuration composed by Hydra.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    logger.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.lightning_model, _recursive_=False)

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(config.get("callbacks"))

    logger.info("Instantiating loggers...")
    lightning_logger: List[Logger] = instantiate_loggers(config.get("logger"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=lightning_logger)

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": lightning_logger,
        "trainer": trainer,
    }

    if lightning_logger:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if config.get("test"):
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def main(config: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Args:
        config: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float]: Optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in config, print config tree, etc.)
    extras(config)

    # train the model
    metric_dict, _ = train(config)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=config.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
