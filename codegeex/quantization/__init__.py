from .quantize import quantize
try:
    from .quantize_oneflow import quantize_oneflow
    from .quantize_oneflow import QuantizedLinear
except ModuleNotFoundError:
    pass
