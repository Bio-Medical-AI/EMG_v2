from typing import Any, Dict, List, Tuple
import os

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils import RankedLogger, extras, instantiate_loggers,log_hyperparameters, task_wrapper


logger = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        config (DictConfig): DictConfig configuration composed by Hydra.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Tuple with metrics and dict with all instantiated objects.
    """
    assert config.checkpoint_path

    logger.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, _recursive_=False)

    logger.info(f"Instantiating model <{config.lightning_model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.lightning_model, _recursive_=False)

    logger.info("Instantiating loggers...")
    lightning_logger: List[Logger] = instantiate_loggers(config.get("logger"))

    logger.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=lightning_logger)

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "logger": lightning_logger,
        "trainer": trainer,
    }

    if lightning_logger:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    logger.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.checkpoint_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=config.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="evaluate")
def main(config: DictConfig) -> None:
    """Main entry point for training.

    Args:
        config: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in config, print config tree, etc.)
    extras(config)

    evaluate(config)


if __name__ == "__main__":
    main()
