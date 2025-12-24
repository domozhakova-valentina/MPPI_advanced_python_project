"""
C++ реализация MPPI через PyBind11
"""
try:
    from .mppi_cpp import MPPICpp
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    MPPICpp = None

__all__ = ['MPPICpp']