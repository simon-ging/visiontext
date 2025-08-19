from __future__ import annotations

import numpy as np
import torch

from visiontext.iotools.feature_compression import (
    NormsC,
    compress_fp32_to_uint8_numpy,
    compress_fp32_to_uint8_torch,
    decompress_uint8_to_fp32_numpy,
    decompress_uint8_to_fp32_torch,
)


def test_feature_compression_numpy():
    generator = np.random.default_rng(0)
    data_randn = generator.standard_normal((10, 10, 10))
    data_rand = generator.uniform(size=(10, 10, 10))

    for norm in NormsC.values():
        for feat, distribution_name in zip(
            (
                data_randn,
                data_rand,
                np.log(np.linspace(0.5, 2.5, 1000)),
                np.linspace(-10, 10, 1000),
            ),
            ("normal", "uniform", "log", "linear"),
        ):
            feat_uint8, mins, maxs = compress_fp32_to_uint8_numpy(feat, norm=norm)
            eps = np.mean(
                (feat - decompress_uint8_to_fp32_numpy(feat_uint8, mins, maxs, norm=norm)) ** 2
            )
            print(f"data {distribution_name} eps {eps:.4e} norm {norm}")
            assert eps < 1e-2


def test_feature_compression_torch():
    data_randn = torch.randn((10, 10, 10))
    data_rand = torch.rand(size=(10, 10, 10))

    for norm in NormsC.values():
        for feat, distribution_name in zip(
            (
                data_randn,
                data_rand,
                torch.log(torch.linspace(0.5, 2.5, 1000)),
                torch.linspace(-10, 10, 1000),
            ),
            ("normal", "uniform", "log", "linear"),
        ):
            feat_uint8, mins, maxs = compress_fp32_to_uint8_torch(feat, norm=norm)
            eps = torch.mean(
                (feat - decompress_uint8_to_fp32_torch(feat_uint8, mins, maxs, norm=norm)) ** 2
            )
            print(f"data {distribution_name} eps {eps:.4e} norm {norm}")
            assert eps < 1e-2
