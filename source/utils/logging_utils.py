from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from utils import pylogger

logger = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Args:
        object_dict (Dict[str, Any]): A dictionary containing the following objects:
        - `"config"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    config = OmegaConf.to_container(object_dict["config"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    hparams["callbacks"] = config.get("callbacks")
    hparams["extras"] = config.get("extras")

    hparams["task_name"] = config.get("task_name")
    hparams["tags"] = config.get("core").get("tags")
    hparams["checkpoint_path"] = config.get("paths").get("checkpoint_path")
    hparams["seed"] = config.get("seed")

    # send hparams to all loggers
    for lightning_logger in trainer.loggers:
        lightning_logger.log_hyperparams(hparams)
