"""
Check whether GPU is available for PyTorch or TensorFlow.

Oneliners:

python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"

Torch backends: https://pytorch.org/docs/stable/backends.html

"""
from packg import format_exception
from visiontext.profiling import get_gpu_profiler, GPUProfilerTorch


def main():
    print(f"\nCheck PyTorch GPU availability")
    try:
        import torch  # noqa

        print(f"    torch.version.__version__={torch.version.__version__}")
        print(f"    torch.version.cuda={torch.version.cuda}")
        print(f"    torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"    torch.cuda.device_count()={torch.cuda.device_count()}")
        print(f"    torch.backends.cudnn.version()={torch.backends.cudnn.version()}")
        print(f"    torch.cuda.get_device_capability()={torch.cuda.get_device_capability()}")
        print(f"    torch.cuda.is_bf16_supported()={torch.cuda.is_bf16_supported()}")
        # https://github.com/pytorch/pytorch/issues/75427
        # before compute capability 8.0 bfloat16 does not offer speed-ups and might fail.
        example_tensor = torch.zeros(128, 1024, 768).cuda()
        print(f"    example_tensor.shape={example_tensor.shape}")

    except Exception as e:
        print(f"    PyTorch failed: {format_exception(e)}")

    print(f"\nCheck TensorFlow GPU availability")
    try:
        from tensorflow.python.client import device_lib  # noqa

        print(device_lib.list_local_devices())
    except Exception as e:
        print(f"    TensorFlow failed: {format_exception(e)}")

    profiler = get_gpu_profiler()
    print(profiler)
    print(f"\nGPU Profiler output: {profiler.profile_to_str()}")


if __name__ == "__main__":
    main()
