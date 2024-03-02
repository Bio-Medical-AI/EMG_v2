import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from utils import pylogger, rich_utils

logger = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(config: DictConfig) -> None:
    """
    Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    Args:
        config (DictConfig): A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not config.get("extras"):
        logger.warning("Extras config not found! <config.extras=null>")
        return

    # disable python warnings
    if config.extras.get("ignore_warnings"):
        logger.info("Disabling python warnings! <config.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if config.extras.get("enforce_tags"):
        logger.info("Enforcing tags! <config.extras.enforce_tags=True>")
        rich_utils.enforce_tags(config, save_to_file=True)

    # pretty print config tree using Rich library
    if config.extras.get("print_config"):
        logger.info("Printing config tree with Rich! <config.extras.print_config=True>")
        rich_utils.print_config_tree(config, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.logger` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    Args:
        task_func (Callable): The task function to be wrapped.

    Returns:
        Callable: The wrapped task function.
    """

    def wrap(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict, object_dict = task_func(config=config)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.logger` file
            logger.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            logger.info(f"Output dir: {config.paths.hydra_output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    logger.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str] = None) -> Optional[float]:
    """Safely retrieves value of the metric logged in LightningModule.


    Args:
        metric_dict (Dict[str, Any]): A dict containing metric values.
        metric_name (Optional[str], optional): If provided, the name of the metric to retrieve. Defaults to None.

    Raises:
        Exception: Exception if metric is not found.

    Returns:
        Optional[float]: If a metric name was provided, the value of the metric.
    """
    if not metric_name:
        logger.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    logger.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
