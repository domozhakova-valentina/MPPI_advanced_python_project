from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

ext_modules = [
    Pybind11Extension(
        "mppi_cpp",
        ["src/mppi/cpp/mppi_cpp.cpp", "src/mppi/cpp/bindings.cpp"],
        include_dirs=["src/mppi/cpp"],
        cxx_std=11,
    ),
]

setup(
    name="mppi_pendulum",
    version="0.1.0",
    author="Domozhakova Valentina",
    description="MPPI для балансировки перевернутого маятника",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)