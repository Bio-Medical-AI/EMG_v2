from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils import pylogger

logger = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_config: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    Args:
        callbacks_config (DictConfig): A DictConfig object containing callback configurations.

    Raises:
        TypeError: If callbacks_config is not a DictConfig object.

    Returns:
        List[Callback]: List of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_config:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_config, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, callback in callbacks_config.items():
        if isinstance(callback, DictConfig) and "_target_" in callback:
            logger.info(f"Instantiating callback <{callback._target_}>")
            callbacks.append(hydra.utils.instantiate(callback))

    return callbacks


def instantiate_loggers(logger_config: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    Args:
        logger_config (DictConfig): A DictConfig object containing logger configurations.

    Raises:
        TypeError: If logger_config is not a DictConfig object.

    Returns:
        List[Logger]: A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    if not logger_config:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lightning_logger in logger_config.items():
        if isinstance(lightning_logger, DictConfig) and "_target_" in lightning_logger:
            logger.info(f"Instantiating logger <{lightning_logger._target_}>")
            loggers.append(hydra.utils.instantiate(lightning_logger))

    return loggers
