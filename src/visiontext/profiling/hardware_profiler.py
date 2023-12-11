import nvidia_smi
import os
import psutil
import pynvml
import torch
from loguru import logger
from typing import Optional, List

from packg import format_exception


def profile_ram():
    """

    Returns:
        RAM total (GB), RAM used (GB)
    """
    mem = psutil.virtual_memory()
    ram_total: float = mem.total / 1024**3
    ram_used: float = mem.used / 1024**3
    return ram_total, ram_used


def get_gpu_profiler():
    try:
        profiler = GPUProfilerNvml()
        return profiler
    except Exception as e:
        logger.error(
            f"Could not setup CUDA NVML GPU Profiler. Falling back to torch profiler. "
            f"Reason: {format_exception(e)}"
        )
        print(f"Log message")
    profiler = GPUProfilerTorch()
    return profiler


class GPUProfilerInterface:
    def profile_gpu(self, gpu_numbers: Optional[List[int]] = None):
        raise NotImplementedError

    def profile_to_str(self, gpu_numbers: Optional[List[int]] = None):
        """
        Use profile output to create a string

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        names, mem_total, mem_used, load_gpu, load_gpu_mem, temp = self.profile_gpu(gpu_numbers)
        ram_total, ram_used = profile_ram()
        # average / sum over all GPUs
        sum_mem_total: float = sum(mem_total)
        sum_mem_used: float = sum(mem_used)
        # gpu_mem_percent: float = gpu_mem_used / gpu_mem_total
        avg_load_gpu: float = sum(load_gpu) / max(1, len(load_gpu))
        avg_load_gpu_mem: float = sum(load_gpu_mem) / max(1, len(load_gpu_mem))
        avg_temp: float = sum(temp) / max(1, len(temp))

        # log the values
        multi_load, multi_load_mem, multi_temp, multi_mem = "", "", "", ""
        if len(load_gpu) > 1:
            multi_load = f' [{" ".join([f"{ld:.0%}" for ld in load_gpu])}]'
            multi_load_mem = f' [{" ".join([f"{ld:.0%}" for ld in load_gpu_mem])}]'
            multi_temp = f' [{" ".join([f"{ld:d}" for ld in temp])}]'
            multi_mem = f' [{" ".join([f"{mem:.1f}GB" for mem in mem_used])}]'

        out_str = " ".join(
            (
                # f"GPU {gpu_names_str}",
                f"Load {avg_load_gpu:.0%}{multi_load}",
                f"UMem {avg_load_gpu_mem:.0%}{multi_load_mem}",
                f"Mem {sum_mem_used:.1f}/{sum_mem_total:.1f}{multi_mem}",
                f"Temp {avg_temp:.0f}°C{multi_temp}",
                f"RAM {ram_used:.1f}/{ram_total:.1f}",
            )
        )

        return out_str

    def get_gpu_numbers(self):
        """
        Try to determine gpu numbers from env variables, fallback is to profile all gpus

        Returns:
            List with GPU numbers
        """
        gpu_count = self.gpu_count  # noqa
        gpu_numbers = list(range(gpu_count))
        device_env = os.getenv("CUDA_VISIBLE_DEVICES")
        if device_env is not None:
            try:
                gpu_numbers = [int(x.strip()) for x in device_env.split(",") if x.strip() != ""]
            except Exception:
                pass
            if len(gpu_numbers) == 0:
                gpu_numbers = list(range(gpu_count))
        return gpu_numbers

    def get_gpu_names(self, gpu_numbers: Optional[List[int]] = None):
        names, mem_total, mem_used, load_gpu, load_gpu_mem, temp = self.profile_gpu(
            gpu_numbers=gpu_numbers
        )
        return names

    def profile_single_gpu(self, gpu_number: int = 0):
        outputs = self.profile_gpu([gpu_number])
        return [x[0] for x in outputs]

    def check_gpus_have_errors(self, gpu_numbers: Optional[List[int]] = None):
        raise NotImplementedError


class GPUProfilerTorch(GPUProfilerInterface):
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count == 0:
            self.gpu_count = 1
        self.has_nvml = True
        try:
            pynvml.nvmlInit()
        except Exception as e:
            logger.error(
                f"Torch profiler cannot use nvml. Utilization and temperature values will not be "
                f"displayed. Reason: {format_exception(e)}"
            )
            self.has_nvml = False

    def profile_gpu(self, gpu_numbers: Optional[List[int]] = None):
        """
        Profile GPU utilization

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            Lists with entries for each GPU with content:
                name, total memory (GB), used memory (GB), gpu load (0-1), memory load (0-1), temperature (°C)

        """
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        devices = [f"cuda:{n}" for n in gpu_numbers]
        props = [torch.cuda.get_device_properties(device) for device in devices]
        mem_total = [p.total_memory / 1024**3 for p in props]
        mem_used = [torch.cuda.max_memory_allocated(device) / 1024**3 for device in devices]
        names = [p.name for p in props]
        if self.has_nvml:
            temp = [torch.cuda.temperature(device) for device in devices]
            load_gpu = [torch.cuda.utilization(device) / 100 for device in devices]
            load_gpu_mem = [torch.cuda.memory_usage(device) for device in devices]
        else:
            temp = [-273] * len(devices)
            load_gpu = [-1] * len(devices)
            load_gpu_mem = [-1] * len(devices)
        # safeguard in case names comes back as bytes (can happen for nvml, not sure for torch)
        names_str = []
        for name in names:
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            names_str.append(name)
        return names_str, mem_total, mem_used, load_gpu, load_gpu_mem, temp

    def check_gpus_have_errors(self, gpu_numbers: Optional[List[int]] = None):
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        has_errors = False
        for i in gpu_numbers:
            try:
                torch.cuda.get_device_properties(f"cuda:{i}")
            except Exception as e:
                logger.error(f"GPU {i} has error: {format_exception(e)}")
                has_errors = True
        return has_errors


class GPUProfilerNvml(GPUProfilerInterface):
    """
    Get information about GPU and RAM usage as string.

    Setup:
        pip install nvidia-ml-py3 pynvml

    Notes:
        gpu_load:
            Percent of time over the past second in which any work has been executing on the GPU.
        gpu_memory_load:
            Percent of time over the past second in which any framebuffer memory has been read or stored.

    Usage:
        >>> gpu_profiler = GPUProfilerNvml()
        >>> print(f"Detected GPUs: {gpu_profiler.get_gpu_names()}")
        >>> print(gpu_profiler.profile_to_str())

    """

    def __init__(self):
        self.gpu_count = -1
        self.gpu_handles = []

        nvidia_smi.nvmlInit()
        self.gpu_count = nvidia_smi.nvmlDeviceGetCount()
        if self.gpu_count == 0:
            self.gpu_count = 1
        self.gpu_handles = []
        for i in range(self.gpu_count):
            try:
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                print(f"WARNING: Could not get handle for GPU {i} due to {format_exception(e)}")
                handle = None
            self.gpu_handles.append(handle)

    def profile_gpu(self, gpu_numbers: Optional[List[int]] = None):
        """
        Profile GPU utilization

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            Lists with entries for each GPU with content:
                name, total memory (GB), used memory (GB), gpu load (0-1), memory load (0-1), temperature (°C)

        """
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers
        gpu_numbers = [i for i in gpu_numbers if self.gpu_handles[i] is not None]

        mem_objs = [nvidia_smi.nvmlDeviceGetMemoryInfo(self.gpu_handles[i]) for i in gpu_numbers]
        mem_total = [mem_obj.total / 1024**3 for mem_obj in mem_objs]
        mem_used = [mem_obj.used / 1024**3 for mem_obj in mem_objs]
        names = [nvidia_smi.nvmlDeviceGetName(self.gpu_handles[i]) for i in gpu_numbers]
        load_objs = [
            nvidia_smi.nvmlDeviceGetUtilizationRates(self.gpu_handles[i]) for i in gpu_numbers
        ]
        load_gpu = [load_obj.gpu / 100 for load_obj in load_objs]
        load_gpu_mem = [load_obj.memory / 100 for load_obj in load_objs]
        temp = [
            nvidia_smi.nvmlDeviceGetTemperature(
                self.gpu_handles[i], nvidia_smi.NVML_TEMPERATURE_GPU
            )
            for i in gpu_numbers
        ]
        names_str = []
        for name in names:
            # on some systems the names come back as bytes, make sure they are string
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            names_str.append(name)
        return names_str, mem_total, mem_used, load_gpu, load_gpu_mem, temp

    def check_gpus_have_errors(self, gpu_numbers: Optional[List[int]] = None):
        """
        Some GPUs can silently not work, and then jax will silently switch to CPU.
        This function checks for these errors.

        Args:
            gpu_numbers: optional numbers of GPUs to profile

        Returns:
            profile string for output
        """
        gpu_numbers = self.get_gpu_numbers() if gpu_numbers is None else gpu_numbers

        has_errors = False
        for i in gpu_numbers:
            if self.gpu_handles[i] is None:
                has_errors = True
                logger.error(f"ERROR: GPU {i} handle could not be created at startup.")
                continue
            try:
                nvidia_smi.nvmlDeviceGetClockInfo(
                    self.gpu_handles[i], nvidia_smi.NVML_CLOCK_GRAPHICS
                )
            except Exception as e:
                logger.error(f"GPU {i} has error: {format_exception(e)}")
                has_errors = True
        return has_errors
