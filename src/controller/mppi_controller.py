<<<<<<< HEAD
"""
Единый интерфейс контроллера (Facade Pattern)
"""
import numpy as np
from typing import Dict, Any, Optional

try:
    # Если запускаем из корня проекта
    from mppi.base import MPPIBase
    from mppi.numpy import MPPINumpy
    from mppi.jax import MPPIJax
    from mppi.cpp import MPPICpp
except ImportError:
    # Если запускаем из поддиректории
    from ..mppi.base import MPPIBase
    from ..mppi.numpy import MPPINumpy
    from ..mppi.jax import MPPIJax
    from ..mppi.cpp import MPPICpp

from .config import MPPIConfig, PendulumConfig


class MPPIController:
    """Фасад для управления различными реализациями MPPI"""
    
    _implementations = {
        'numpy': MPPINumpy,
        'jax': MPPIJax,
        'cpp': MPPICpp
    }
    
    def __init__(self, 
                 pendulum_config: PendulumConfig,
                 mppi_config: MPPIConfig,
                 implementation: str = 'numpy'):
        """
        Args:
            pendulum_config: конфигурация маятника
            mppi_config: конфигурация алгоритма MPPI
            implementation: 'numpy', 'jax' или 'cpp'
        """
        self.pendulum_config = pendulum_config
        self.mppi_config = mppi_config
        self.implementation_name = implementation
        
        # Создание выбранной реализации
        if implementation not in self._implementations:
            raise ValueError(f"Недопустимая реализация: {implementation}")
        
        # Объединение конфигураций для MPPI
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        
        try:
            self.mppi: MPPIBase = self._implementations[implementation](combined_config)
        except ImportError as e:
            raise ImportError(f"Не удалось загрузить {implementation}: {e}")
        
        # Состояние системы
        self.state = np.array([
            pendulum_config.x0,
            pendulum_config.theta0,
            pendulum_config.dx0,
            pendulum_config.dtheta0
        ])
        
        # История
        self.history = {
            'states': [],
            'controls': [],
            'costs': []
        }
    
    def step(self) -> Dict[str, Any]:
        """
        Выполнить один шаг управления
        
        Returns:
            Словарь с результатами шага
        """
        # Вычисление управления
        force = self.mppi.compute_control(self.state)
        
        # Ограничение силы
        force = np.clip(force, 
                       -self.pendulum_config.max_force,
                       self.pendulum_config.max_force)
        
        # Интегрирование динамики (метод Эйлера)
        derivatives = self.mppi._dynamics(self.state, force)
        self.state = self.state + derivatives * self.pendulum_config.dt
        
        # Ограничение положения тележки
        self.state[0] = np.clip(self.state[0],
                               -self.pendulum_config.max_x,
                               self.pendulum_config.max_x)
        
        # Сохранение истории
        self.history['states'].append(self.state.copy())
        self.history['controls'].append(force)
        self.history['costs'].append(self.mppi.costs_history[-1] 
                                     if self.mppi.costs_history else 0.0)
        
        return {
            'state': self.state.copy(),
            'force': force,
            'cost': self.history['costs'][-1]
        }
    
    def reset(self):
        """Сброс системы и контроллера"""
        self.state = np.array([
            self.pendulum_config.x0,
            self.pendulum_config.theta0,
            self.pendulum_config.dx0,
            self.pendulum_config.dtheta0
        ])
        self.mppi.reset()
        self.history = {'states': [], 'controls': [], 'costs': []}
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Вычисление метрик производительности"""
        if not self.history['states']:
            return {}
        
        states = np.array(self.history['states'])
        controls = np.array(self.history['controls'])
        
        return {
            'avg_angle': np.mean(np.abs(states[:, 1])),
            'max_angle': np.max(np.abs(states[:, 1])),
            'avg_force': np.mean(np.abs(controls)),
            'total_cost': np.sum(self.history['costs'])
        }


class CombinedConfig:
    """Объединённая конфигурация для MPPI"""
    def __init__(self, pendulum: PendulumConfig, mppi: MPPIConfig):
        self.m_cart = pendulum.m_cart
        self.m_pole = pendulum.m_pole
        self.l = pendulum.l
        self.g = pendulum.g
        self.dt = pendulum.dt
        
        self.K = mppi.K
        self.T = mppi.T
        self.lambda_ = mppi.lambda_
        self.sigma = mppi.sigma
        self.Q = mppi.Q
        self.R = mppi.R
=======
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Callable, Union
from enum import Enum
import json
from pathlib import Path
import warnings
from datetime import datetime
import threading
import queue

from .config import (
    SystemConfig,
    MPPIConfig,
    State,
    SimulationConfig,
    ExperimentConfig,
    Implementation,
    CostComponent,
    get_preset_config,
    create_default_configs
)

from ..mppi import create_controller as create_backend_controller
from ..mppi import get_available_implementations


class ControllerStatus(Enum):
    """Статус контроллера"""
    IDLE = "idle"  # Ожидание
    RUNNING = "running"  # Выполняется
    PAUSED = "paused"  # Приостановлен
    STOPPED = "stopped"  # Остановлен
    ERROR = "error"  # Ошибка


@dataclass
class PerformanceMetrics:
    """Метрики производительности

    Паттерн: Value Object
    """
    # Временные метрики
    total_time: float = 0.0  # общее время работы (с)
    avg_step_time: float = 0.0  # среднее время шага (с)
    min_step_time: float = 0.0  # минимальное время шага (с)
    max_step_time: float = 0.0  # максимальное время шага (с)
    fps: float = 0.0  # кадров в секунду

    # Метрики управления
    avg_control: float = 0.0  # среднее управление (Н)
    control_variance: float = 0.0  # дисперсия управления
    max_control: float = 0.0  # максимальное управление (Н)
    control_effort: float = 0.0  # усилие управления (интеграл F²)

    # Метрики состояния
    avg_angle: float = 0.0  # средний угол (рад)
    max_angle: float = 0.0  # максимальный угол (рад)
    settling_time: float = 0.0  # время установления (с)
    overshoot: float = 0.0  # перерегулирование (%)

    # Метрики стоимости
    avg_cost: float = 0.0  # средняя стоимость
    min_cost: float = 0.0  # минимальная стоимость
    max_cost: float = 0.0  # максимальная стоимость
    total_cost: float = 0.0  # общая стоимость

    # Метрики успеха
    success: bool = False  # успешное выполнение
    success_criteria: Dict[str, Any] = field(default_factory=dict)  # критерии успеха

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            'time': {
                'total': self.total_time,
                'avg_step': self.avg_step_time,
                'min_step': self.min_step_time,
                'max_step': self.max_step_time,
                'fps': self.fps
            },
            'control': {
                'avg': self.avg_control,
                'variance': self.control_variance,
                'max': self.max_control,
                'effort': self.control_effort
            },
            'state': {
                'avg_angle': self.avg_angle,
                'max_angle': self.max_angle,
                'settling_time': self.settling_time,
                'overshoot': self.overshoot
            },
            'cost': {
                'avg': self.avg_cost,
                'min': self.min_cost,
                'max': self.max_cost,
                'total': self.total_cost
            },
            'success': {
                'success': self.success,
                'criteria': self.success_criteria
            }
        }

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"PerformanceMetrics(\n"
                f"  Time: {self.avg_step_time * 1000:.1f}ms/step, "
                f"{self.fps:.0f} FPS\n"
                f"  Control: avg={self.avg_control:.2f}N, "
                f"max={self.max_control:.2f}N\n"
                f"  State: angle={math.degrees(self.avg_angle):.1f}°, "
                f"settling={self.settling_time:.2f}s\n"
                f"  Success: {self.success}\n"
                f")")


@dataclass
class SimulationResult:
    """Результат симуляции

    Паттерн: Value Object
    """
    # Конфигурации
    system_config: SystemConfig
    mppi_config: MPPIConfig
    simulation_config: SimulationConfig

    # Данные траектории
    time_steps: List[float] = field(default_factory=list)
    states: List[State] = field(default_factory=list)
    controls: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    compute_times: List[float] = field(default_factory=list)

    # Метаданные
    implementation: str = "numpy"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = False
    error_message: Optional[str] = None

    # Метрики
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    def add_step(self, time_step: float, state: State, control: float,
                 cost: float, compute_time: float):
        """Добавляет шаг симуляции"""
        self.time_steps.append(time_step)
        self.states.append(state)
        self.controls.append(control)
        self.costs.append(cost)
        self.compute_times.append(compute_time)

    def get_final_state(self) -> Optional[State]:
        """Возвращает конечное состояние"""
        if self.states:
            return self.states[-1]
        return None

    def get_duration(self) -> float:
        """Возвращает продолжительность симуляции"""
        if self.time_steps:
            return self.time_steps[-1]
        return 0.0

    def get_num_steps(self) -> int:
        """Возвращает количество шагов"""
        return len(self.time_steps)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            'metadata': {
                'implementation': self.implementation,
                'timestamp': self.timestamp,
                'success': self.success,
                'error_message': self.error_message,
                'num_steps': self.get_num_steps(),
                'duration': self.get_duration()
            },
            'configs': {
                'system': self.system_config.to_dict(),
                'mppi': self.mppi_config.to_dict(),
                'simulation': self.simulation_config.to_dict()
            },
            'data': {
                'time_steps': self.time_steps,
                'states': [s.to_dict() for s in self.states],
                'controls': self.controls,
                'costs': self.costs,
                'compute_times': self.compute_times
            },
            'metrics': self.metrics.to_dict()
        }

    def save(self, filename: str):
        """Сохраняет результат в файл"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'SimulationResult':
        """Загружает результат из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Загружаем конфигурации
        system_config = SystemConfig.from_dict(data['configs']['system'])
        mppi_config = MPPIConfig.from_dict(data['configs']['mppi'])
        simulation_config = SimulationConfig.from_dict(data['configs']['simulation'])

        # Создаем результат
        result = cls(
            system_config=system_config,
            mppi_config=mppi_config,
            simulation_config=simulation_config,
            implementation=data['metadata']['implementation'],
            timestamp=data['metadata']['timestamp'],
            success=data['metadata']['success'],
            error_message=data['metadata']['error_message']
        )

        # Загружаем данные
        result_data = data['data']
        for i in range(len(result_data['time_steps'])):
            state = State.from_dict(result_data['states'][i])
            result.add_step(
                result_data['time_steps'][i],
                state,
                result_data['controls'][i],
                result_data['costs'][i],
                result_data['compute_times'][i]
            )

        # Загружаем метрики
        if 'metrics' in data:
            metrics_data = data['metrics']
            result.metrics = PerformanceMetrics(
                total_time=metrics_data['time']['total'],
                avg_step_time=metrics_data['time']['avg_step'],
                min_step_time=metrics_data['time']['min_step'],
                max_step_time=metrics_data['time']['max_step'],
                fps=metrics_data['time']['fps'],
                avg_control=metrics_data['control']['avg'],
                control_variance=metrics_data['control']['variance'],
                max_control=metrics_data['control']['max'],
                control_effort=metrics_data['control']['effort'],
                avg_angle=metrics_data['state']['avg_angle'],
                max_angle=metrics_data['state']['max_angle'],
                settling_time=metrics_data['state']['settling_time'],
                overshoot=metrics_data['state']['overshoot'],
                avg_cost=metrics_data['cost']['avg'],
                min_cost=metrics_data['cost']['min'],
                max_cost=metrics_data['cost']['max'],
                total_cost=metrics_data['cost']['total'],
                success=metrics_data['success']['success'],
                success_criteria=metrics_data['success']['criteria']
            )

        return result


class MPPIController:
    """Единый интерфейс контроллера MPPI

    Паттерн: Facade - предоставляет простой интерфейс для работы
              с различными реализациями алгоритма MPPI
    """

    def __init__(self, implementation: Union[str, Implementation] = "numpy",
                 system_config: Optional[SystemConfig] = None,
                 mppi_config: Optional[MPPIConfig] = None,
                 simulation_config: Optional[SimulationConfig] = None,
                 **kwargs):
        """Инициализирует контроллер

        Args:
            implementation: реализация (numpy, jax, cpp)
            system_config: конфигурация системы
            mppi_config: конфигурация алгоритма
            simulation_config: конфигурация симуляции
            **kwargs: дополнительные аргументы для конкретной реализации
        """
        # Преобразуем строку в Enum если нужно
        if isinstance(implementation, str):
            implementation = Implementation.from_string(implementation)

        self.implementation = implementation
        self.implementation_name = implementation.name.lower()

        # Конфигурации
        self.system_config = system_config or SystemConfig()
        self.mppi_config = mppi_config or MPPIConfig()
        self.simulation_config = simulation_config or SimulationConfig()

        # Проверяем доступность реализации
        available_impls = get_available_implementations()
        if self.implementation_name not in available_impls:
            raise ValueError(f"Реализация {self.implementation_name} недоступна")
        if not available_impls[self.implementation_name]:
            raise ImportError(f"Реализация {self.implementation_name} не установлена")

        # Создаем бэкенд контроллер
        self._backend = create_backend_controller(
            self.implementation_name,
            self.system_config,
            self.mppi_config,
            **kwargs
        )

        # Состояние контроллера
        self.status = ControllerStatus.IDLE
        self.current_state = self.simulation_config.initial_state.copy()
        self.current_time = 0.0
        self.iteration = 0

        # История
        self.history = {
            'time': [],
            'states': [],
            'controls': [],
            'costs': [],
            'compute_times': []
        }

        # Подписчики на события
        self._observers = {
            'step_completed': [],
            'simulation_started': [],
            'simulation_completed': [],
            'error_occurred': []
        }

        # Блокировка для потокобезопасности
        self._lock = threading.RLock()

        print(f"Создан MPPIController с реализацией {self.implementation_name.upper()}")

    def add_observer(self, event: str, callback: Callable):
        """Добавляет наблюдателя за событиями

        Паттерн: Observer

        Args:
            event: событие (step_completed, simulation_started и т.д.)
            callback: функция обратного вызова
        """
        if event in self._observers:
            self._observers[event].append(callback)

    def remove_observer(self, event: str, callback: Callable):
        """Удаляет наблюдателя"""
        if event in self._observers and callback in self._observers[event]:
            self._observers[event].remove(callback)

    def _notify_observers(self, event: str, *args, **kwargs):
        """Уведомляет наблюдателей о событии"""
        if event in self._observers:
            for callback in self._observers[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    print(f"Ошибка в обработчике события {event}: {e}")

    def compute_control(self, state: Optional[State] = None) -> float:
        """Вычисляет управляющее воздействие для текущего состояния

        Args:
            state: состояние (если None, используется текущее состояние)

        Returns:
            управляющее воздействие (сила)
        """
        with self._lock:
            if self.status == ControllerStatus.ERROR:
                raise RuntimeError("Контроллер в состоянии ошибки")

            if state is None:
                state = self.current_state

            try:
                start_time = time.perf_counter()

                # Вычисляем управление с помощью бэкенда
                control = self._backend.compute_control(state)

                compute_time = time.perf_counter() - start_time

                # Вычисляем стоимость
                cost = self._compute_step_cost(state, control)

                # Сохраняем в историю
                self.history['time'].append(self.current_time)
                self.history['states'].append(state.copy())
                self.history['controls'].append(control)
                self.history['costs'].append(cost)
                self.history['compute_times'].append(compute_time)

                # Уведомляем наблюдателей
                step_data = {
                    'time': self.current_time,
                    'state': state,
                    'control': control,
                    'cost': cost,
                    'compute_time': compute_time,
                    'iteration': self.iteration
                }
                self._notify_observers('step_completed', step_data)

                self.iteration += 1

                return control

            except Exception as e:
                self.status = ControllerStatus.ERROR
                self._notify_observers('error_occurred', e)
                raise

    def _compute_step_cost(self, state: State, control: float) -> float:
        """Вычисляет стоимость для одного шага"""
        cost = 0.0

        # Штраф за угол
        cost += self.mppi_config.cost_weights.angle * state.theta ** 2

        # Штраф за угловую скорость
        cost += self.mppi_config.cost_weights.angular_velocity * state.theta_dot ** 2

        # Штраф за положение
        cost += self.mppi_config.cost_weights.position * state.x ** 2

        # Штраф за скорость
        cost += self.mppi_config.cost_weights.velocity * state.x_dot ** 2

        # Штраф за управление
        cost += self.mppi_config.cost_weights.control * control ** 2

        # Штраф за скорость изменения управления (если есть предыдущее управление)
        if self.history['controls']:
            prev_control = self.history['controls'][-1]
            control_rate = control - prev_control
            cost += self.mppi_config.cost_weights.control_rate * control_rate ** 2

        return cost

    def step_simulation(self, control: Optional[float] = None) -> State:
        """Выполняет один шаг симуляции

        Args:
            control: управление (если None, вычисляется)

        Returns:
            новое состояние
        """
        with self._lock:
            if control is None:
                control = self.compute_control(self.current_state)

            # Обновляем состояние с помощью модели бэкенда
            # (предполагаем, что у бэкенда есть метод step в модели)
            next_state = self._backend.model.step(
                self.current_state,
                control,
                self.system_config.dt
            )

            # Обновляем время и состояние
            self.current_time += self.system_config.dt
            self.current_state = next_state

            return next_state

    def run_simulation(self, config: Optional[SimulationConfig] = None,
                       callback: Optional[Callable] = None) -> SimulationResult:
        """Запускает полную симуляцию

        Args:
            config: конфигурация симуляции
            callback: функция обратного вызова после каждого шага

        Returns:
            результат симуляции
        """
        with self._lock:
            if self.status == ControllerStatus.RUNNING:
                raise RuntimeError("Симуляция уже выполняется")

            # Обновляем конфигурацию если предоставлена
            if config is not None:
                self.simulation_config = config

            # Сбрасываем состояние
            self.reset()

            # Устанавливаем начальное состояние
            self.current_state = self.simulation_config.initial_state.copy()
            self.current_time = 0.0

            # Создаем результат
            result = SimulationResult(
                system_config=self.system_config.copy(),
                mppi_config=self.mppi_config.copy(),
                simulation_config=self.simulation_config.copy(),
                implementation=self.implementation_name
            )

            # Запускаем симуляцию
            self.status = ControllerStatus.RUNNING
            self._notify_observers('simulation_started')

            try:
                num_steps = int(self.simulation_config.duration / self.system_config.dt)

                for step in range(num_steps):
                    # Проверяем статус
                    if self.status != ControllerStatus.RUNNING:
                        break

                    # Вычисляем управление
                    start_time = time.perf_counter()
                    control = self.compute_control(self.current_state)
                    compute_time = time.perf_counter() - start_time

                    # Вычисляем стоимость
                    cost = self._compute_step_cost(self.current_state, control)

                    # Сохраняем шаг
                    result.add_step(
                        self.current_time,
                        self.current_state.copy(),
                        control,
                        cost,
                        compute_time
                    )

                    # Обновляем состояние
                    self.step_simulation(control)

                    # Вызываем callback
                    if callback:
                        callback(step, num_steps, self.current_state, control, cost)

                    # Выводим прогресс
                    if self.simulation_config.print_progress and step % 100 == 0:
                        print(f"Шаг {step}/{num_steps}, "
                              f"Время: {self.current_time:.2f}s, "
                              f"Угол: {math.degrees(self.current_state.theta):.1f}°")

                # Вычисляем метрики
                result.metrics = self._compute_metrics(result)
                result.success = result.metrics.success

                self.status = ControllerStatus.IDLE
                self._notify_observers('simulation_completed', result)

                return result

            except Exception as e:
                self.status = ControllerStatus.ERROR
                result.error_message = str(e)
                self._notify_observers('error_occurred', e)
                raise

    def _compute_metrics(self, result: SimulationResult) -> PerformanceMetrics:
        """Вычисляет метрики производительности"""
        if not result.states:
            return PerformanceMetrics()

        # Временные метрики
        compute_times = result.compute_times
        total_time = sum(compute_times)
        avg_step_time = total_time / len(compute_times) if compute_times else 0
        min_step_time = min(compute_times) if compute_times else 0
        max_step_time = max(compute_times) if compute_times else 0
        fps = 1.0 / avg_step_time if avg_step_time > 0 else 0

        # Метрики управления
        controls = result.controls
        avg_control = np.mean(np.abs(controls)) if controls else 0
        control_variance = np.var(controls) if controls else 0
        max_control = np.max(np.abs(controls)) if controls else 0
        control_effort = np.sum(np.array(controls) ** 2) * self.system_config.dt

        # Метрики состояния
        angles = [s.theta for s in result.states]
        avg_angle = np.mean(np.abs(angles)) if angles else 0
        max_angle = np.max(np.abs(angles)) if angles else 0

        # Время установления (когда угол становится меньше 5 градусов)
        settling_threshold = math.radians(5.0)
        settling_time = 0.0
        for i, state in enumerate(result.states):
            if abs(state.theta) < settling_threshold:
                settling_time = result.time_steps[i]
                break

        # Перерегулирование
        overshoot = 0.0
        if angles:
            max_deviation = np.max(np.abs(angles))
            initial_angle = abs(result.states[0].theta)
            if initial_angle > 0:
                overshoot = (max_deviation - initial_angle) / initial_angle * 100

        # Метрики стоимости
        costs = result.costs
        avg_cost = np.mean(costs) if costs else 0
        min_cost = np.min(costs) if costs else 0
        max_cost = np.max(costs) if costs else 0
        total_cost = np.sum(costs) if costs else 0

        # Критерии успеха
        success = (max_angle < math.radians(45.0) and  # Угол меньше 45 градусов
                   settling_time < self.simulation_config.duration * 0.5)  # Установление за половину времени

        success_criteria = {
            'max_angle_deg': math.degrees(max_angle),
            'max_angle_limit_deg': 45.0,
            'settling_time': settling_time,
            'settling_time_limit': self.simulation_config.duration * 0.5
        }

        return PerformanceMetrics(
            total_time=total_time,
            avg_step_time=avg_step_time,
            min_step_time=min_step_time,
            max_step_time=max_step_time,
            fps=fps,
            avg_control=avg_control,
            control_variance=control_variance,
            max_control=max_control,
            control_effort=control_effort,
            avg_angle=avg_angle,
            max_angle=max_angle,
            settling_time=settling_time,
            overshoot=overshoot,
            avg_cost=avg_cost,
            min_cost=min_cost,
            max_cost=max_cost,
            total_cost=total_cost,
            success=success,
            success_criteria=success_criteria
        )

    def reset(self):
        """Сбрасывает контроллер в начальное состояние"""
        with self._lock:
            self._backend.reset()
            self.current_state = self.simulation_config.initial_state.copy()
            self.current_time = 0.0
            self.iteration = 0
            self.history = {
                'time': [],
                'states': [],
                'controls': [],
                'costs': [],
                'compute_times': []
            }
            self.status = ControllerStatus.IDLE

    def pause(self):
        """Приостанавливает симуляцию"""
        with self._lock:
            if self.status == ControllerStatus.RUNNING:
                self.status = ControllerStatus.PAUSED

    def resume(self):
        """Возобновляет симуляцию"""
        with self._lock:
            if self.status == ControllerStatus.PAUSED:
                self.status = ControllerStatus.RUNNING

    def stop(self):
        """Останавливает симуляцию"""
        with self._lock:
            if self.status in [ControllerStatus.RUNNING, ControllerStatus.PAUSED]:
                self.status = ControllerStatus.STOPPED

    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус контроллера"""
        with self._lock:
            return {
                'status': self.status.value,
                'iteration': self.iteration,
                'current_time': self.current_time,
                'current_state': self.current_state.to_dict(),
                'history_size': len(self.history['states'])
            }

    def get_backend_stats(self) -> Dict[str, Any]:
        """Возвращает статистику бэкенда"""
        try:
            return self._backend.get_stats()
        except AttributeError:
            return {}

    def save(self, filename: str):
        """Сохраняет состояние контроллера"""
        with self._lock:
            data = {
                'implementation': self.implementation_name,
                'system_config': self.system_config.to_dict(),
                'mppi_config': self.mppi_config.to_dict(),
                'simulation_config': self.simulation_config.to_dict(),
                'current_state': self.current_state.to_dict(),
                'current_time': self.current_time,
                'iteration': self.iteration,
                'status': self.status.value,
                'history': self.history
            }

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, filename: str) -> 'MPPIController':
        """Загружает состояние контроллера"""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Создаем конфигурации
        system_config = SystemConfig.from_dict(data['system_config'])
        mppi_config = MPPIConfig.from_dict(data['mppi_config'])
        simulation_config = SimulationConfig.from_dict(data['simulation_config'])

        # Создаем контроллер
        controller = cls(
            implementation=data['implementation'],
            system_config=system_config,
            mppi_config=mppi_config,
            simulation_config=simulation_config
        )

        # Восстанавливаем состояние
        controller.current_state = State.from_dict(data['current_state'])
        controller.current_time = data['current_time']
        controller.iteration = data['iteration']
        controller.status = ControllerStatus(data['status'])
        controller.history = data['history']

        return controller

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"MPPIController(\n"
                f"  implementation: {self.implementation_name},\n"
                f"  status: {self.status.value},\n"
                f"  iteration: {self.iteration},\n"
                f"  state: {self.current_state}\n"
                f")")


class MPPIManager:
    """Менеджер для управления несколькими контроллерами и сравнения

    Паттерн: Mediator - координирует работу нескольких контроллеров
    """

    def __init__(self):
        """Инициализирует менеджер"""
        self.controllers: Dict[str, MPPIController] = {}
        self.results: Dict[str, SimulationResult] = {}
        self.comparisons: Dict[str, Dict[str, Any]] = {}

    def create_controller(self, name: str, implementation: str = "numpy",
                          **kwargs) -> MPPIController:
        """Создает новый контроллер

        Args:
            name: имя контроллера
            implementation: реализация
            **kwargs: аргументы для контроллера

        Returns:
            созданный контроллер
        """
        if name in self.controllers:
            raise ValueError(f"Контроллер с именем '{name}' уже существует")

        controller = MPPIController(implementation=implementation, **kwargs)
        self.controllers[name] = controller

        # Добавляем наблюдателя для автоматического сохранения результатов
        def save_result_callback(result):
            self.results[name] = result

        controller.add_observer('simulation_completed', save_result_callback)

        return controller

    def run_simulation(self, name: str, config: Optional[SimulationConfig] = None,
                       **kwargs) -> SimulationResult:
        """Запускает симуляцию для контроллера

        Args:
            name: имя контроллера
            config: конфигурация симуляции
            **kwargs: дополнительные аргументы

        Returns:
            результат симуляции
        """
        if name not in self.controllers:
            raise ValueError(f"Контроллер '{name}' не найден")

        controller = self.controllers[name]
        result = controller.run_simulation(config, **kwargs)
        self.results[name] = result

        return result

    def run_comparison(self, config: Optional[SimulationConfig] = None,
                       implementations: Optional[List[str]] = None) -> Dict[str, Any]:
        """Запускает сравнение различных реализаций

        Args:
            config: конфигурация симуляции
            implementations: список реализаций для сравнения

        Returns:
            результаты сравнения
        """
        if implementations is None:
            implementations = ["numpy", "jax", "cpp"]

        comparison_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_results = {
            'id': comparison_id,
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict() if config else None,
            'results': {}
        }

        for impl in implementations:
            try:
                # Создаем контроллер
                controller_name = f"{impl}_{comparison_id}"
                controller = self.create_controller(controller_name, impl)

                # Запускаем симуляцию
                result = self.run_simulation(controller_name, config)

                # Сохраняем результаты
                comparison_results['results'][impl] = {
                    'metrics': result.metrics.to_dict(),
                    'success': result.success,
                    'num_steps': result.get_num_steps(),
                    'duration': result.get_duration()
                }

            except Exception as e:
                comparison_results['results'][impl] = {
                    'error': str(e),
                    'success': False
                }

        self.comparisons[comparison_id] = comparison_results
        return comparison_results

    def get_comparison_summary(self, comparison_id: str) -> Dict[str, Any]:
        """Возвращает сводку сравнения

        Args:
            comparison_id: ID сравнения

        Returns:
            сводка сравнения
        """
        if comparison_id not in self.comparisons:
            raise ValueError(f"Сравнение с ID '{comparison_id}' не найдено")

        comparison = self.comparisons[comparison_id]
        summary = {
            'id': comparison_id,
            'timestamp': comparison['timestamp'],
            'num_implementations': len(comparison['results']),
            'implementations': list(comparison['results'].keys()),
            'successful': [],
            'failed': [],
            'performance': {}
        }

        for impl, result in comparison['results'].items():
            if 'error' in result:
                summary['failed'].append(impl)
            else:
                summary['successful'].append(impl)
                if 'metrics' in result:
                    summary['performance'][impl] = {
                        'avg_step_time': result['metrics']['time']['avg_step'],
                        'fps': result['metrics']['time']['fps'],
                        'settling_time': result['metrics']['state']['settling_time'],
                        'success': result['success']
                    }

        return summary

    def save_comparison(self, comparison_id: str, filename: str):
        """Сохраняет сравнение в файл"""
        if comparison_id not in self.comparisons:
            raise ValueError(f"Сравнение с ID '{comparison_id}' не найдено")

        with open(filename, 'w') as f:
            json.dump(self.comparisons[comparison_id], f, indent=2)

    def load_comparison(self, filename: str) -> str:
        """Загружает сравнение из файла"""
        with open(filename, 'r') as f:
            comparison = json.load(f)

        comparison_id = comparison['id']
        self.comparisons[comparison_id] = comparison
        return comparison_id

    def get_controller(self, name: str) -> MPPIController:
        """Возвращает контроллер по имени"""
        if name not in self.controllers:
            raise ValueError(f"Контроллер '{name}' не найден")
        return self.controllers[name]

    def remove_controller(self, name: str):
        """Удаляет контроллер"""
        if name in self.controllers:
            del self.controllers[name]
        if name in self.results:
            del self.results[name]

    def clear(self):
        """Очищает все контроллеры и результаты"""
        self.controllers.clear()
        self.results.clear()
        self.comparisons.clear()

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"MPPIManager(\n"
                f"  controllers: {len(self.controllers)},\n"
                f"  results: {len(self.results)},\n"
                f"  comparisons: {len(self.comparisons)}\n"
                f")")


# Функции высокого уровня

def create_controller(implementation: str = "numpy", **kwargs) -> MPPIController:
    """Создает контроллер MPPI

    Args:
        implementation: реализация (numpy, jax, cpp)
        **kwargs: аргументы для контроллера

    Returns:
        созданный контроллер
    """
    return MPPIController(implementation=implementation, **kwargs)


def benchmark_implementations(config: Optional[SimulationConfig] = None,
                              implementations: Optional[List[str]] = None,
                              num_runs: int = 3) -> Dict[str, Any]:
    """Запускает бенчмарк различных реализаций

    Args:
        config: конфигурация симуляции
        implementations: список реализаций
        num_runs: количество запусков для каждой реализации

    Returns:
        результаты бенчмарка
    """
    if implementations is None:
        implementations = ["numpy", "jax", "cpp"]

    if config is None:
        config = SimulationConfig(duration=2.0)  # Короткая симуляция для бенчмарка

    manager = MPPIManager()
    results = {}

    for impl in implementations:
        try:
            run_times = []
            for run in range(num_runs):
                controller_name = f"{impl}_benchmark_{run}"
                controller = manager.create_controller(controller_name, impl)

                start_time = time.time()
                result = manager.run_simulation(controller_name, config)
                end_time = time.time()

                run_times.append(end_time - start_time)

            results[impl] = {
                'available': True,
                'avg_time': np.mean(run_times),
                'std_time': np.std(run_times),
                'min_time': np.min(run_times),
                'max_time': np.max(run_times),
                'all_times': run_times
            }

        except Exception as e:
            results[impl] = {
                'available': False,
                'error': str(e)
            }

    return results


def run_experiment(experiment_config: ExperimentConfig) -> Dict[str, Any]:
    """Запускает эксперимент с варьированием параметров

    Args:
        experiment_config: конфигурация эксперимента

    Returns:
        результаты эксперимента
    """
    results = {
        'experiment_name': experiment_config.name,
        'timestamp': datetime.now().isoformat(),
        'config': experiment_config.to_dict(),
        'runs': []
    }

    # Создаем все комбинации параметров
    param_combinations = []

    if not experiment_config.varying_parameters:
        # Без варьирования параметров
        param_combinations.append({})
    else:
        # Генерируем все комбинации
        param_values = []
        param_names = []

        for param_range in experiment_config.varying_parameters:
            param_names.append(param_range.name)
            param_values.append(param_range.get_values())

        # Рекурсивно генерируем комбинации
        def generate_combinations(current, idx):
            if idx == len(param_values):
                param_combinations.append(current.copy())
                return

            for value in param_values[idx]:
                current[param_names[idx]] = value
                generate_combinations(current, idx + 1)

        generate_combinations({}, 0)

    # Запускаем эксперименты
    for param_combo in param_combinations:
        for run in range(experiment_config.num_runs):
            try:
                # Создаем конфигурации с текущими параметрами
                system_config = experiment_config.fixed_system_config.copy()
                mppi_config = experiment_config.fixed_mppi_config.copy()

                # Применяем параметры
                for param_name, param_value in param_combo.items():
                    # Определяем, к какой конфигурации относится параметр
                    if hasattr(system_config, param_name):
                        setattr(system_config, param_name, param_value)
                    elif hasattr(mppi_config, param_name):
                        setattr(mppi_config, param_name, param_value)
                    elif hasattr(mppi_config.cost_weights, param_name):
                        setattr(mppi_config.cost_weights, param_name, param_value)

                # Создаем и запускаем контроллер
                controller = create_controller(
                    implementation="numpy",  # Используем numpy для экспериментов
                    system_config=system_config,
                    mppi_config=mppi_config
                )

                # Запускаем симуляцию
                sim_config = SimulationConfig(
                    duration=5.0,
                    initial_angle_deg=20.0
                )

                result = controller.run_simulation(sim_config)

                # Сохраняем результаты
                run_result = {
                    'parameters': param_combo,
                    'run': run,
                    'success': result.success,
                    'metrics': result.metrics.to_dict(),
                    'system_config': system_config.to_dict(),
                    'mppi_config': mppi_config.to_dict()
                }

                results['runs'].append(run_result)

            except Exception as e:
                run_result = {
                    'parameters': param_combo,
                    'run': run,
                    'success': False,
                    'error': str(e)
                }
                results['runs'].append(run_result)

    return results


def save_controller(controller: MPPIController, filename: str):
    """Сохраняет контроллер в файл"""
    controller.save(filename)


def load_controller(filename: str) -> MPPIController:
    """Загружает контроллер из файла"""
    return MPPIController.load(filename)
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
