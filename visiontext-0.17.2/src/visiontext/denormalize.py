"""
"""

import torch
import torchvision
import torchvision.transforms.v2
from packg.log import logger


class Denormalize:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        # print(f"Created normalizer with shape {self.mean.shape} {self.std.shape} ")

    def __call__(self, image_tensor: torch.Tensor):
        """
        Args:
            image_tensor: either (C, H, W) or (B, C, H, W)

        Returns:
            same image_tensor but with normalization undone, for human viewing

        """
        # print(f"Got input with shape {tensor.shape} {tensor.dtype}")
        # normalize op is: output[channel] = (input[channel] - mean[channel]) / std[channel]
        # therefore denormalize is: output[channel] = input[channel] * std[channel] + mean[channel]
        new_dims = [1] * len(image_tensor.shape)

        if image_tensor.ndim == 4:
            new_dims[1] = -1
        elif image_tensor.ndim == 3:
            new_dims[0] = -1
        else:
            raise ValueError(f"Expected 3 or 4 dims, got shape {image_tensor.shape}")
        mean = self.mean.view(*new_dims).to(image_tensor.device)
        std = self.std.view(*new_dims).to(image_tensor.device)
        # print(f"mean {mean.shape} {mean.dtype} std {std.shape} {std.dtype}")
        return image_tensor * std + mean

    @classmethod
    def from_transforms(cls, transforms):
        mean, std = find_mean_and_std_for_denormalization(transforms)
        if mean is None or std is None:
            logger.warning(f"Could not find mean and std for denormalization from {transforms=}")
            mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        return cls(mean, std)


def find_mean_and_std_for_denormalization(transforms):
    mean, std = None, None

    # call recursively if it is a list of transforms
    transform_list = None
    if isinstance(transforms, list):
        transform_list = transforms
    elif isinstance(
        transforms, (torchvision.transforms.Compose, torchvision.transforms.v2.Compose)
    ):
        transform_list = transforms.transforms
    if transform_list is not None:
        for tf in transform_list:
            mean, std = find_mean_and_std_for_denormalization(tf)
            if mean is not None and std is not None:
                break
        return mean, std

    if hasattr(transforms, "mean") and hasattr(transforms, "std"):
        mean, std = transforms.mean, transforms.std
    return mean, std
