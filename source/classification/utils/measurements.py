import numpy as np
import torch
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress
from torch import nn


def measure_eval_time(model: nn.Module, dummy_input: torch.Tensor) -> tuple[float, float]:
    """
    Measure the time of evaluating model or module.
    Args:
        model: Any model/classifier
        dummy_input: tensor of size accepted by given model

    Returns:
        tuple of mean time, measured in milliseconds and its standard deviation
    """
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    dummy_input = dummy_input.to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(100):
            _ = model(dummy_input)
        # MEASURE PERFORMANCE
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings).item()
    return mean_syn, std_syn


def calculate_mean_std(
    file_paths: list[str], progress: CustomProgress | None = None
) -> tuple[float, float]:
    """
    Measure mean and std of data in iterative manner
    Args:
        file_paths: Any model/classifier
        progress: Optional progress object for creation of progress bar

    Returns:
        tuple of mean and std
    """
    total_sum = 0
    total_count = 0

    if progress is not None:
        task = progress.add_task("[cyan]Calculating Mean", total=len(file_paths))
    for file_path in file_paths:
        arr = np.load(file_path)
        total_sum += np.sum(arr)
        total_count += np.size(arr)
        if progress is not None:
            progress.update(task, advance=1)
    mean = total_sum / total_count
    total_sum = 0
    if progress is not None:
        task = progress.add_task("[cyan]Calculating Std and Std", total=len(file_paths))
    for file_path in file_paths:
        arr = np.load(file_path)
        total_sum += np.sum((arr - mean) ** 2)
        if progress is not None:
            progress.update(task, advance=1)
    std_deviation = np.sqrt(total_sum / total_count)

    return mean, std_deviation
