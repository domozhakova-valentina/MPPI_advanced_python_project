import sys
import os

# Имя файла в зависимости от ОС
if sys.platform == "win32":
    lib_name = "mppi_cpp.pyd"
else:
    lib_name = "mppi_cpp.so"

lib_path = os.path.join(os.path.dirname(__file__), lib_name)

if os.path.exists(lib_path):
    try:
        # Прямой импорт скомпилированного модуля
        import importlib.util
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
    print("ℹC++ модуль не скомпилирован. Запустите build_cpp.py")

__all__ = ['MPPICpp']