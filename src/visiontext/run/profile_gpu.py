"""
Check whether GPU is available for PyTorch or TensorFlow.

Oneliners:

python -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.version.cuda)"

Torch backends: https://pytorch.org/docs/stable/backends.html

"""
from packg import format_exception


def main():
    # print(GPUProfiler().profile_to_str())  # todo add gpu profiler and enable this line

    print(f"Check PyTorch GPU availability")
    try:
        import torch  # noqa

        print(f"torch.version.__version__={torch.version.__version__}")
        print(f"torch.version.cuda={torch.version.cuda}")
        print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        print(f"torch.backends.cudnn.version()={torch.backends.cudnn.version()}")
    except Exception as e:
        print(f"PyTorch failed: {format_exception(e)}")

    print(f"Check TensorFlow GPU availability")
    try:
        from tensorflow.python.client import device_lib  # noqa

        print(device_lib.list_local_devices())
    except Exception as e:
        print(f"TensorFlow failed: {format_exception(e)}")

        return


if __name__ == "__main__":
    main()
