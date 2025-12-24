import sys
import os
import glob

# Найти .pyd или .so в текущей папке
files = glob.glob(os.path.join(os.path.dirname(__file__), "mppi_cpp*.pyd")) \
        if sys.platform == "win32" else \
        glob.glob(os.path.join(os.path.dirname(__file__), "mppi_cpp*.so"))

if files:
    try:
        import importlib.util
        lib_path = files[0]  # берём первый подходящий
        spec = importlib.util.spec_from_file_location("mppi_cpp", lib_path)
        mppi_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mppi_module)
        MPPICpp = mppi_module.MPPICpp
        CPP_AVAILABLE = True
    except Exception as e:
        CPP_AVAILABLE = False
        MPPICpp = None
        print(f"Не удалось загрузить C++ модуль: {e}")
else:
    CPP_AVAILABLE = False
    MPPICpp = None
    print("ℹC++ модуль не найден. Сборка отсутствует или не завершена")

__all__ = ['MPPICpp']
