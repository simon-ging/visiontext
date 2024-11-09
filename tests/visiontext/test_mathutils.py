import numpy as np
import pytest
import torch

from visiontext.mathutils import torch_stable_softmax


@pytest.mark.parametrize(
    "inp, temp, expected_result",
    [
        (torch.tensor([1.0, 2.0, 3.0]), 1.0, torch.tensor([0.09003057, 0.24472847, 0.66524096])),
        (torch.tensor([1.0, 2.0, 3.0]), 2.0, torch.tensor([0.186324, 0.307196, 0.50648])),
        (torch.tensor([1000.0, 2000.0, 3000.0]), 1.0, torch.tensor([0.0, 0.0, 1.0])),
        (torch.tensor([-1.0, -2.0, -3.0]), 1.0, torch.tensor([0.66524096, 0.24472847, 0.09003057])),
    ],
)
def test_stable_softmax(inp, temp, expected_result):
    result = torch_stable_softmax(inp, temp=temp)
    np.testing.assert_allclose(result, expected_result, rtol=1e-5)


if __name__ == "__main__":
    pytest.main()
