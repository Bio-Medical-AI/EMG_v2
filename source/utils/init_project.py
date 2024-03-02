from typing import Dict, List

import os
from pathlib import Path


def _export_environs(environs: Dict[str, str]) -> None:
    """Exports all environmental variables for Hydra to use.

    Args:
        environs (Dict[str, str]): Dict of environmental variables.
    """
    for k, v in environs.items():
        os.environ[k] = str(v)


def _make_dirs(dirs: List[str]) -> None:
    """Creates the directories from the list if not exist.

    Args:
        dirs (List[str]): List of directories to create.
    """
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)


def init_project(root: str) -> None:
    """Inits the project environmental variables. This is later used by Hydra.

    Args:
        root (str): Name of the calling module.
    """
    project_name = Path(root).parent.name

    environs = {}

    # project root
    environs['PROJECT_ROOT'] = Path(__file__).parent.parent.parent.resolve()
    environs['STORAGE_DIR'] = environs['PROJECT_ROOT'] / "storage"
    environs['DATA_DIR'] = environs['STORAGE_DIR'] / "data"
    environs['PROJECT_STORAGE_DIR'] = environs['STORAGE_DIR'] / project_name
    environs['PROJECT_DIR'] = environs['PROJECT_ROOT'] / "source" / project_name
    environs['CONFIG_DIR'] = environs['PROJECT_DIR'] / "config"

    # build directory for the storage and project
    _make_dirs([environs['STORAGE_DIR'], environs['DATA_DIR'], environs['PROJECT_STORAGE_DIR']])

    # export env vars
    _export_environs(environs)
