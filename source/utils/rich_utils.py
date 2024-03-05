from pathlib import Path
from typing import Sequence

import numpy as np
import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning import Trainer
from lightning.pytorch.callbacks.progress.rich_progress import (
    CustomInfiniteTask,
    CustomProgress,
    RichProgressBar,
)
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
from utils import pylogger

logger = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        config (DictConfig): A DictConfig composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
            Defaults to ( "datamodule", "model", "callbacks", "logger", "trainer", "paths", "extras").
        resolve (bool, optional): hether to resolve reference fields of DictConfig. Defaults to False.
        save_to_file (bool, optional): Whether to export config to the hydra output folder. Defaults to False.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (
            queue.append(field)
            if field in config
            else logger.warning(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in config:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(config.paths.run_dir, "config_tree.logger"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(config: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    Args:
        config (DictConfig): A DictConfig composed by Hydra.
        save_to_file (bool): Whether to export tags to the hydra output folder. Defaults to False.
    """
    if not config.get("core").get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        logger.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(config):
            config.core.tags = tags

        logger.info(f"Tags: {config.tags}")

    if save_to_file:
        with open(Path(config.paths.run_dir, "tags.logger"), "w") as file:
            rich.print(config.core.tags, file=file)


def get_progress_callback(trainer: Trainer) -> CustomProgress | None:
    """Extract progress object from Progress Callback from Trainer. Progress might not be
    initialized yet if you are extracting it at the wrong stage of pipeline. Progress bars
    initialized by this function may not load properly.

    Args:
        trainer (Trainer): Trainer from which to extract progress callback.

    Returns:
        Progress object that could be used for creating progress bars.
    """
    if trainer is None:
        return None
    calls = [callback for callback in trainer.callbacks if isinstance(callback, RichProgressBar)]
    if len(calls) > 0:
        return calls[0].progress
    return None


def progress_load(path: str, task: CustomInfiniteTask, progress: CustomProgress) -> np.ndarray:
    """Extract progress object from Progress Callback from Trainer. Progress might not be
    initialized yet if you are extracting it at the wrong stage of pipeline. Progress bars
    initialized by this function may not load properly.

    Args:
        path (str): Trainer from which to extract progress callback.
        task (CustomInfiniteTask): Trainer from which to extract progress callback.
        progress (CustomProgress): Trainer from which to extract progress callback.

    Returns:
        Progress object that could be used for creating progress bars.
    """
    progress.update(task, advance=1)
    return np.load(path)
