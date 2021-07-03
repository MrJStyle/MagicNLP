from enum import Enum


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    XPU = "xpu"
