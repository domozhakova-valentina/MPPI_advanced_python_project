import subprocess
import platform
import shutil
from pathlib import Path
import sys


def build_cpp():
    """Сборка C++ расширения с помощью CMake"""
    
    # Определяем пути
    project_dir = Path(__file__).parent
    src_cpp_dir = project_dir / "src" / "mppi" / "cpp"
    build_dir = project_dir / "build"
    
    # Создаем директорию для сборки
    build_dir.mkdir(exist_ok=True)
    
    # Команда для CMake
    system = platform.system()

    # Путь к Cmake
    cmake_path = shutil.which("cmake")
    
    if system == "Windows":
        # Для Windows
        pybind11_cmake_dir = subprocess.check_output(
             ["python", "-m", "pybind11", "--cmakedir"], text=True).strip()

        cmd_cmake = [
            cmake_path,
            "-G", "Ninja",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
            ".."
        ]
        cmd_build = [cmake_path, "--build", "."]
    else:
        # Для Linux/Mac
        cmd_cmake = [cmake_path, ".."]
        cmd_build = ["make"]
    
    # Запускаем CMake
    subprocess.run(cmd_cmake, cwd=build_dir, check=True)
    
    # Собираем проект
    subprocess.run(cmd_build, cwd=build_dir, check=True)

    print("Сборка завершена успешно")
        

if __name__ == "__main__":
    build_cpp()