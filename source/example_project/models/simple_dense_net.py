import os

import hydra
import torch
from torch import nn
from omegaconf import DictConfig


class SimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 784,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 10,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        Args:

            input_size (int): The number of input features.
            lin1_size (int): The number of output features of the first linear layer.
            lin2_size (int): The number of output features of the second linear layer.
            lin3_size (int): The number of output features of the third linear layer.
            output_size (int): The number of output features of the final linear layer.
        """
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output
        """
        batch_size, _, _, _ = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _run(config: DictConfig) -> None:
    """
    Run to test if the module works.
    """
    model = hydra.utils.instantiate(config.model, _recursive_=False)
    print(f'{model}')


if __name__ == "__main__":
    _run()
