import torch
from torchvision.transforms import Compose
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2 import RandomResizedCrop

__all__ = [
    "Compose",
    "Flip",
    "NormalizeMinMax",
    "RandomResizedCrop",
    "Resize",
    "SetRandomPoissonExposure",
]


class Flip:
    """Flip data along the vertical axis"""

    def __init__(self, p=1.0) -> None:
        self.p = p

    def __call__(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        if torch.rand(1) <= self.p:
            return torch.fliplr(sample)
        else:
            return sample

    def __repr__(self) -> str:
        s = f"Flip sample horizontally"
        if self.p < 1.0:
            s += f" with {self.p:.0%} chance"
        return s


class NormalizeMinMax:
    """Standardize range of an array to have min 0 and max 1

    Returns:
        the standardized data
    """

    def __init__(self, min: float = 0, max: float = 1) -> None:
        self.min = min
        self.max = max

    def __call__(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        out = (sample - sample.min()) / (sample.max() - sample.min())
        return out * (self.max - self.min) + self.min

    def __repr__(self) -> str:
        return f"Normalized to [0,1] range"


class Resize:
    """Resize image to a given shape

    Args:
        output_shape: the target shape
    """

    def __init__(self, output_shape: int | tuple | list, second_dim: int = None) -> None:
        if isinstance(output_shape, list):
            output_shape = tuple(output_shape)
        elif isinstance(output_shape, int) and isinstance(second_dim, int):
            output_shape = (output_shape, second_dim)
        if not isinstance(output_shape, (int, tuple)):
            raise ValueError(f"output_shape must be int or tuple, not {type(output_shape)}")
        self.output_shape = tuple(output_shape)

    def __call__(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        # ensure sample has shape (...,h,w)
        if len(sample.shape) == 2:
            sample = sample.unsqueeze(0)
        return (
            TF.resize(sample, self.output_shape).squeeze().float()
        )  # (...,h,w) -> (...,*output_shape)

    def __repr__(self):
        return f"Resize to {self.output_shape}"


class SetRandomPoissonExposure:
    """Set exposure of an image to a fixed or random value

    Simulate the effect of varying exposure time on an image by setting the exposure to
    roughly the same number of counts. The effect is equivalent to adding Poisson noise,
    and can be used to simulate ARPES data with varying exposure times.

    Args:
        low: the minimum value of the exposure. if None, set to 1e3. if high is None,
            this is the value used for the exposure.
        high (optional): the maximum value of the exposure. if None, set to low
        normalize: whether to normalize the image to the range [0,1] before setting
            exposure
    """

    def __init__(
        self, low: int | float = 1e3, high: int | float | None = None, normalize=True
    ) -> None:
        self.low = torch.tensor([low])
        self.high = high
        if self.high is not None:
            self.high = torch.tensor([high])
        self.normalize = normalize

    def __call__(self, sample) -> torch.Tensor:
        if self.normalize:
            sample = sample - sample.min()
            sample = sample / sample.sum()
        if self.high is None:
            counts = self.low
        else:
            counts = torch.randint(low=self.low, high=self.high, size=(1,)).to(sample.device)
        exposed = torch.poisson(sample * counts)
        return exposed.to(sample.device)

    def __repr__(self) -> str:
        return f"Set exposure to random range ({self.low}, {self.high})"
