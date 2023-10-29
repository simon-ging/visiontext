from __future__ import annotations

import numpy as np

from visiontext.iotools.feature_compression import feat_to_uint8, uint8_to_feat


def test_feat_to_uint8():
    generator = np.random.default_rng(0)
    data_randn = generator.standard_normal((10, 10, 10))
    data_rand = generator.uniform(size=(10, 10, 10))

    for norm in "linear", "exp":
        for feat, distribution_name in zip(
            (
                data_randn,
                data_rand,
                np.log(np.linspace(0.5, 2.5, 1000)),
                np.linspace(-10, 10, 1000),
            ),
            ("normal", "uniform", "log", "linear"),
        ):
            feat_uint8, mins, maxs = feat_to_uint8(feat, norm=norm)
            eps = np.mean((feat - uint8_to_feat(feat_uint8, mins, maxs, norm=norm)) ** 2)
            print(f"data {distribution_name} eps {eps:.4e} norm {norm}")
            assert eps < 1e-2
