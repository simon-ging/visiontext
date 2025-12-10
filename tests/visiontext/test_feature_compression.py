from __future__ import annotations

import numpy as np
import pytest
import torch

from visiontext.iotools.feature_compression import (
    NormsC,
    compress_fp32_to_uint8_numpy,
    compress_fp32_to_uint8_torch,
    convert_to_fp16_torch,
    decompress_uint8_to_fp32_numpy,
    decompress_uint8_to_fp32_torch,
    dump_safetensors_zst,
    dump_single_safetensor_zst,
    load_safetensors_zst,
    load_single_safetensor_zst,
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


def test_convert_to_fp16_torch_success():
    """Test successful conversion to fp16 with valid values."""
    # Create a tensor with values well within fp16 range
    feat = torch.randn((10, 10)) * 100  # Max abs ~300-400
    feat_fp16 = convert_to_fp16_torch(feat)
    assert feat_fp16.dtype == torch.float16
    # for fp16 we have to accept some loss in precision
    torch.testing.assert_close(feat, feat_fp16.float(), atol=0.1, rtol=1e-3)


def test_convert_to_fp16_torch_overflow():
    """Test that conversion raises error when values exceed fp16 max."""
    # Create a tensor with values that exceed fp16 max (65504)
    feat = torch.tensor([70000.0, 1.0, -70000.0])

    with pytest.raises(ValueError, match="overflow fp16"):
        convert_to_fp16_torch(feat)


def test_convert_to_fp16_torch_edge_case():
    """Test conversion at the edge of fp16 range."""
    # fp16 max is 65504
    feat = torch.tensor([65499.0, -65499.0, 0.0])
    feat_fp16 = convert_to_fp16_torch(feat)
    assert feat_fp16.dtype == torch.float16


def test_dump_load_safetensors_zst_basic(tmp_path):
    """Test basic save and load of safetensors with zstd compression."""
    # Create test tensors
    tensors = {
        "tensor1": torch.randn((5, 10)),
        "tensor2": torch.rand((3, 7)) * 100,
        "tensor3": torch.ones((2, 2)),
    }

    save_path = tmp_path / "test_tensors.safetensors.zst"

    # Save
    dump_safetensors_zst(tensors, save_path)
    assert save_path.exists()

    # Load
    loaded_tensors = load_safetensors_zst(save_path)

    # Verify
    assert set(loaded_tensors.keys()) == set(tensors.keys())
    for key in tensors.keys():
        torch.testing.assert_close(tensors[key], loaded_tensors[key])


def test_dump_safetensors_zst_create_parent(tmp_path):
    """Test saving with create_parent=True."""
    tensors = {"data": torch.randn((5, 5))}
    save_path = tmp_path / "subdir" / "nested" / "test.safetensors.zst"

    # Should create parent directories
    dump_safetensors_zst(tensors, save_path, create_parent=True)
    assert save_path.exists()
    assert save_path.parent.exists()

    # Verify we can load it back
    loaded_tensors = load_safetensors_zst(save_path)
    torch.testing.assert_close(tensors["data"], loaded_tensors["data"])


def test_load_safetensors_zst_file_not_found(tmp_path):
    """Test that loading non-existent file raises appropriate error."""
    nonexistent_path = tmp_path / "does_not_exist.safetensors.zst"

    with pytest.raises(FileNotFoundError):
        load_safetensors_zst(nonexistent_path)


def test_dump_safetensors_zst_with_fp16(tmp_path):
    """Test saving and loading fp16 tensors."""
    # Create tensors with values within fp16 range
    tensors = {
        "fp32_tensor": torch.randn((3, 4)) * 100,
        "fp16_tensor": torch.randn((3, 4)).half(),
    }

    save_path = tmp_path / "mixed_dtype.safetensors.zst"

    dump_safetensors_zst(tensors, save_path)
    loaded_tensors = load_safetensors_zst(save_path)

    # Verify dtypes are preserved
    assert loaded_tensors["fp32_tensor"].dtype == torch.float32
    assert loaded_tensors["fp16_tensor"].dtype == torch.float16

    # Verify values
    torch.testing.assert_close(tensors["fp32_tensor"], loaded_tensors["fp32_tensor"])
    torch.testing.assert_close(tensors["fp16_tensor"], loaded_tensors["fp16_tensor"])


def test_dump_load_single_safetensor_zst(tmp_path):
    """Test single tensor save and load."""
    tensor = torch.randn((4, 8)) * 50
    save_path = tmp_path / "single.safetensors.zst"

    dump_single_safetensor_zst(tensor, save_path)
    loaded_tensor = load_single_safetensor_zst(save_path)

    torch.testing.assert_close(tensor, loaded_tensor)


def test_load_single_safetensor_zst_wrong_keys(tmp_path):
    """Test that load_single_safetensor_zst raises error for multiple tensors."""
    tensors = {"a": torch.randn((2, 2)), "b": torch.randn((3, 3))}
    save_path = tmp_path / "multi.safetensors.zst"

    dump_safetensors_zst(tensors, save_path)

    with pytest.raises(KeyError, match="single 'tensor'"):
        load_single_safetensor_zst(save_path)
