<<<<<<< HEAD
"""
Контроллер для управления маятником
"""
from .config import MPPIConfig, PendulumConfig
from .mppi_controller import MPPIController

__all__ = ['MPPIConfig', 'PendulumConfig', 'MPPIController']
=======
from .config import (
    SystemConfig,
    MPPIConfig,
    State,
    CostWeights,
    SimulationConfig,
    ExperimentConfig,
    ParameterRange,
    ConfigBuilder
)

from .mppi_controller import (
    MPPIController,
    MPPIManager,
    SimulationResult,
    PerformanceMetrics,
    create_controller,
    benchmark_implementations,
    run_experiment,
    load_controller,
    save_controller
)

__all__ = [
    # Конфигурации
    'SystemConfig',
    'MPPIConfig',
    'State',
    'CostWeights',
    'SimulationConfig',
    'ExperimentConfig',
    'ParameterRange',
    'ConfigBuilder',

    # Контроллер и менеджер
    'MPPIController',
    'MPPIManager',
    'SimulationResult',
    'PerformanceMetrics',

    # Функции
    'create_controller',
    'benchmark_implementations',
    'run_experiment',
    'load_controller',
    'save_controller'
]

__version__ = '1.0.0'


def get_available_backends():
    """Возвращает список доступных бэкендов"""
    from ..mppi import get_available_implementations
    return get_available_implementations()


def print_system_info():
    """Выводит информацию о системе и доступных реализациях"""
    import platform
    import sys
    import numpy as np

    print("=" * 60)
    print("MPPI Controller System Information")
    print("=" * 60)

    # Информация о системе
    print("\nSystem:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version}")
    print(f"  NumPy: {np.__version__}")

    # Информация о доступных бэкендах
    backends = get_available_backends()
    print("\nAvailable Backends:")
    for backend, available in backends.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {backend.upper():<10} {status}")

    # Информация о производительности
    print("\nPerformance Notes:")
    print("  • NumPy: Best for understanding and debugging")
    print("  • JAX: Best for GPU acceleration and gradient-based optimization")
    print("  • C++: Best for maximum CPU performance")

    print("=" * 60)


# Автоматически выводим информацию при импорте
print_system_info()
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
