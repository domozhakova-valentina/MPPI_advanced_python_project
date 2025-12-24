<<<<<<< HEAD
"""
MPPI алгоритм для балансировки перевёрнутого маятника
"""
from .base import MPPIBase
from .numpy import MPPINumpy
from .jax import MPPIJax

__all__ = ['MPPIBase', 'MPPINumpy', 'MPPIJax', 'MPPICpp']
=======
from .base import MPPIBase, DynamicsModel
from .numpy import MPPIController as NumPyMPPI
from .jax import MPPIController as JAXMPPI
try:
    from .cpp import MPPIController as CppMPPI
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

# Функции:
def create_controller(mode: str, system_config: SystemConfig = None,
                      mppi_config: MPPIConfig = None, **kwargs):
    """Фабричный метод для создания контроллера MPPI

    Паттерн: Factory Method

    Args:
        mode: режим реализации ('numpy', 'jax', 'cpp')
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        **kwargs: дополнительные аргументы для конкретной реализации

    Returns:
        экземпляр контроллера MPPI

    Raises:
        ValueError: если указан неподдерживаемый режим
        ImportError: если C++ реализация недоступна
    """
    if system_config is None:
        system_config = SystemConfig()

    if mppi_config is None:
        mppi_config = MPPIConfig()

    if mode == MPPMode.NUMPY:
        from .numpy import MPPIController
        return MPPIController(system_config, mppi_config, **kwargs)

    elif mode == MPPMode.JAX:
        from .jax import MPPIController
        return MPPIController(system_config, mppi_config, **kwargs)

    elif mode == MPPMode.CPP:
        if not CPP_AVAILABLE:
            raise ImportError(
                "C++ реализация не доступна. "
                "Убедитесь, что модуль скомпилирован и установлен."
            )
        from .cpp import MPPIController
        return MPPIController(system_config, mppi_config, **kwargs)

    else:
        raise ValueError(f"Неподдерживаемый режим: {mode}. "
                         f"Доступные режимы: {list(MPPMode.__dict__.values())}")


def get_available_implementations():
    """Возвращает список доступных реализаций MPPI

    Returns:
        словарь с доступными реализациями и их статусом
    """
    implementations = {
        MPPMode.NUMPY: True,
        MPPMode.JAX: True,
        MPPMode.CPP: CPP_AVAILABLE
    }

    return implementations


def benchmark_all_implementations(system_config: SystemConfig = None,
                                  mppi_config: MPPIConfig = None,
                                  test_steps: int = 100):
    """Запускает бенчмарк для всех доступных реализаций

    Args:
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        test_steps: количество шагов для тестирования

    Returns:
        словарь с результатами бенчмарка
    """
    if system_config is None:
        system_config = SystemConfig()

    if mppi_config is None:
        mppi_config = MPPIConfig()

    results = {}
    implementations = get_available_implementations()

    for mode, available in implementations.items():
        if not available:
            results[mode] = {'status': 'unavailable', 'time_per_step': None}
            continue

        try:
            controller = create_controller(mode, system_config, mppi_config)
            state = State(x=0.0, theta=0.1)  # небольшое начальное отклонение

            # Разогрев (warm-up)
            for _ in range(10):
                controller.compute_control(state)

            # Измерение производительности
            import time
            times = []

            for _ in range(test_steps):
                start_time = time.perf_counter()
                controller.compute_control(state)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            results[mode] = {
                'status': 'success',
                'time_per_step': avg_time,
                'total_time': sum(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
            }

        except Exception as e:
            results[mode] = {
                'status': 'error',
                'error': str(e),
                'time_per_step': None
            }

    return results


def compare_implementations(system_config: SystemConfig = None,
                            mppi_config: MPPIConfig = None,
                            initial_state: State = None,
                            simulation_steps: int = 200):
    """Сравнивает работу всех реализаций на одной задаче

    Args:
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        initial_state: начальное состояние
        simulation_steps: количество шагов симуляции

    Returns:
        словарь с результатами сравнения
    """
    if system_config is None:
        system_config = SystemConfig()

    if mppi_config is None:
        mppi_config = MPPIConfig()

    if initial_state is None:
        initial_state = State(x=0.0, theta=0.2, x_dot=0.0, theta_dot=0.0)  # 20 градусов

    results = {}
    implementations = get_available_implementations()

    for mode, available in implementations.items():
        if not available:
            results[mode] = {'status': 'unavailable', 'trajectory': [], 'controls': []}
            continue

        try:
            controller = create_controller(mode, system_config, mppi_config)

            # Сброс контроллера
            controller.reset()

            # Запуск симуляции
            trajectory = []
            controls = []
            costs = []
            state = initial_state

            import time
            start_time = time.perf_counter()

            for step in range(simulation_steps):
                control = controller.compute_control(state)

                # Сохраняем данные
                trajectory.append({
                    'x': float(state.x),
                    'theta': float(state.theta),
                    'x_dot': float(state.x_dot),
                    'theta_dot': float(state.theta_dot),
                    'step': step
                })
                controls.append(float(control))

                # Обновляем состояние
                state = controller.model.step(state, control, system_config.dt)

                # Вычисляем стоимость
                cost = controller.compute_cost([state], [control])
                costs.append(float(cost))

            end_time = time.perf_counter()

            results[mode] = {
                'status': 'success',
                'trajectory': trajectory,
                'controls': controls,
                'costs': costs,
                'total_time': end_time - start_time,
                'avg_time_per_step': (end_time - start_time) / simulation_steps,
                'final_state': {
                    'x': float(state.x),
                    'theta': float(state.theta),
                    'x_dot': float(state.x_dot),
                    'theta_dot': float(state.theta_dot)
                },
                'controller': controller
            }

        except Exception as e:
            results[mode] = {
                'status': 'error',
                'error': str(e),
                'trajectory': [],
                'controls': []
            }

    return results


# Классы:
class MPPMode:
    """Режимы работы MPPI"""
    NUMPY = 'numpy'
    JAX = 'jax'
    CPP = 'cpp'
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
