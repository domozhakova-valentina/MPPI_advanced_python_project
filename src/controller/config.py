from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
from enum import Enum, auto
import json
import yaml
import numpy as np
from pathlib import Path
import math
from copy import deepcopy


class Implementation(Enum):
    """Доступные реализации алгоритма"""
    NUMPY = auto()
    JAX = auto()
    CPP = auto()

    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_string(cls, s: str) -> 'Implementation':
        """Создает из строки"""
        return cls[s.upper()]


class CostComponent(Enum):
    """Компоненты функции стоимости"""
    ANGLE = "angle"  # Штраф за угол отклонения
    ANGULAR_VELOCITY = "angular_velocity"  # Штраф за угловую скорость
    POSITION = "position"  # Штраф за положение тележки
    VELOCITY = "velocity"  # Штраф за скорость тележки
    CONTROL = "control"  # Штраф за управление
    CONTROL_RATE = "control_rate"  # Штраф за скорость изменения управления
    TERMINAL = "terminal"  # Терминальная стоимость


@dataclass
class CostWeights:
    """Веса компонентов функции стоимости

    Паттерн: Value Object
    """
    angle: float = 10.0  # Штраф за угол (θ²)
    angular_velocity: float = 0.1  # Штраф за угловую скорость (θ̇²)
    position: float = 1.0  # Штраф за положение (x²)
    velocity: float = 0.1  # Штраф за скорость (ẋ²)
    control: float = 0.01  # Штраф за управление (F²)
    control_rate: float = 0.001  # Штраф за скорость изменения управления (ΔF²)
    terminal: float = 10.0  # Терминальный штраф

    def to_dict(self) -> Dict[str, float]:
        """Конвертирует в словарь"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'CostWeights':
        """Создает из словаря"""
        return cls(**data)

    def get_weight(self, component: CostComponent) -> float:
        """Возвращает вес для компонента"""
        mapping = {
            CostComponent.ANGLE: self.angle,
            CostComponent.ANGULAR_VELOCITY: self.angular_velocity,
            CostComponent.POSITION: self.position,
            CostComponent.VELOCITY: self.velocity,
            CostComponent.CONTROL: self.control,
            CostComponent.CONTROL_RATE: self.control_rate,
            CostComponent.TERMINAL: self.terminal
        }
        return mapping.get(component, 0.0)


@dataclass
class SystemConfig:
    """Конфигурация системы (маятник + тележка)

    Паттерн: Value Object
    """
    # Основные параметры
    cart_mass: float = 1.0  # M - масса тележки (кг)
    pole_mass: float = 0.1  # m - масса маятника (кг)
    pole_length: float = 1.0  # l - длина маятника (м)
    gravity: float = 9.81  # g - ускорение свободного падения (м/с²)

    # Параметры трения
    friction_cart: float = 0.1  # коэффициент трения тележки
    friction_pole: float = 0.01  # коэффициент трения маятника

    # Параметры симуляции
    dt: float = 0.02  # шаг времени (с)
    simulation_frequency: float = 50.0  # частота симуляции (Гц)
    control_frequency: float = 50.0  # частота управления (Гц)

    # Ограничения
    cart_position_limit: float = 2.0  # ограничение положения тележки (м)
    pole_angle_limit: float = math.pi / 2  # ограничение угла маятника (рад)

    # Параметры шума (для реалистичности)
    measurement_noise: float = 0.001  # шум измерений
    process_noise: float = 0.0001  # шум процесса

    def __post_init__(self):
        """Валидация конфигурации"""
        self._validate()

    def _validate(self):
        """Проверяет корректность конфигурации"""
        if self.cart_mass <= 0:
            raise ValueError("Масса тележки должна быть положительной")
        if self.pole_mass <= 0:
            raise ValueError("Масса маятника должна быть положительной")
        if self.pole_length <= 0:
            raise ValueError("Длина маятника должна быть положительной")
        if self.gravity <= 0:
            raise ValueError("Ускорение свободного падения должно быть положительным")
        if self.dt <= 0:
            raise ValueError("Шаг времени должен быть положительным")
        if self.simulation_frequency <= 0:
            raise ValueError("Частота симуляции должна быть положительной")
        if self.control_frequency <= 0:
            raise ValueError("Частота управления должна быть положительной")
        if self.cart_position_limit <= 0:
            raise ValueError("Ограничение положения тележки должно быть положительным")
        if self.pole_angle_limit <= 0:
            raise ValueError("Ограничение угла маятника должно быть положительным")

    def copy(self) -> 'SystemConfig':
        """Создает копию конфигурации"""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Создает из словаря"""
        return cls(**data)

    def save(self, filename: str):
        """Сохраняет конфигурацию в файл"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'SystemConfig':
        """Загружает конфигурацию из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"SystemConfig(\n"
                f"  cart_mass={self.cart_mass} kg,\n"
                f"  pole_mass={self.pole_mass} kg,\n"
                f"  pole_length={self.pole_length} m,\n"
                f"  gravity={self.gravity} m/s²,\n"
                f"  dt={self.dt} s\n"
                f")")


@dataclass
class MPPIConfig:
    """Конфигурация алгоритма MPPI

    Паттерн: Value Object
    """
    # Основные параметры MPPI
    num_samples: int = 1000  # K - количество траекторий
    horizon: int = 30  # T - горизонт планирования
    lambda_: float = 0.1  # λ - параметр для вычисления весов
    noise_sigma: float = 1.0  # σ - стандартное отклонение шума
    control_limit: float = 10.0  # максимальное значение силы (Н)

    # Параметры температуры
    temperature: float = 1.0  # температура для softmax
    adaptive_temperature: bool = False  # адаптивная температура
    min_temperature: float = 0.1  # минимальная температура
    max_temperature: float = 10.0  # максимальная температура

    # Веса стоимости
    cost_weights: CostWeights = field(default_factory=CostWeights)

    # Параметры шума
    noise_distribution: str = "normal"  # распределение шума (normal, uniform)
    noise_correlation: float = 0.5  # корреляция шума во времени
    noise_adaptive: bool = False  # адаптивный шум

    # Параметры обновления
    learning_rate: float = 1.0  # скорость обучения (обычно 1.0 для MPPI)
    momentum: float = 0.0  # момент для обновления траектории
    update_strategy: str = "shift"  # стратегия обновления (shift, rolling)

    # Расширенные параметры
    use_terminal_cost: bool = True  # использовать терминальную стоимость
    use_state_constraints: bool = True  # использовать ограничения состояния
    use_control_smoothing: bool = True  # сглаживание управления

    # Параметры для параллельных вычислений
    batch_size: int = 100  # размер батча для параллельной обработки
    use_vectorization: bool = True  # использовать векторизацию

    def __post_init__(self):
        """Валидация конфигурации"""
        self._validate()

    def _validate(self):
        """Проверяет корректность конфигурации"""
        if self.num_samples <= 0:
            raise ValueError("Количество траекторий должно быть положительным")
        if self.horizon <= 0:
            raise ValueError("Горизонт планирования должен быть положительным")
        if self.lambda_ <= 0:
            raise ValueError("Параметр lambda должен быть положительным")
        if self.noise_sigma <= 0:
            raise ValueError("Стандартное отклонение шума должно быть положительным")
        if self.control_limit <= 0:
            raise ValueError("Ограничение управления должно быть положительным")
        if self.temperature <= 0:
            raise ValueError("Температура должна быть положительной")
        if self.min_temperature <= 0 or self.max_temperature <= 0:
            raise ValueError("Минимальная и максимальная температура должны быть положительными")
        if self.min_temperature > self.max_temperature:
            raise ValueError("Минимальная температура не может быть больше максимальной")
        if self.noise_distribution not in ["normal", "uniform"]:
            raise ValueError("Распределение шума должно быть 'normal' или 'uniform'")
        if not 0 <= self.noise_correlation <= 1:
            raise ValueError("Корреляция шума должна быть в диапазоне [0, 1]")
        if not 0 <= self.momentum <= 1:
            raise ValueError("Момент должен быть в диапазоне [0, 1]")
        if self.update_strategy not in ["shift", "rolling"]:
            raise ValueError("Стратегия обновления должна быть 'shift' или 'rolling'")

    def copy(self) -> 'MPPIConfig':
        """Создает копию конфигурации"""
        return deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        data = asdict(self)
        # Конвертируем CostWeights в словарь
        data['cost_weights'] = self.cost_weights.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPPIConfig':
        """Создает из словаря"""
        # Извлекаем cost_weights отдельно
        cost_weights_data = data.pop('cost_weights', {})
        cost_weights = CostWeights.from_dict(cost_weights_data)

        # Создаем конфигурацию
        config = cls(**data)
        config.cost_weights = cost_weights
        return config

    def save(self, filename: str):
        """Сохраняет конфигурацию в файл"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'MPPIConfig':
        """Загружает конфигурацию из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_cost_weight(self, component: CostComponent) -> float:
        """Возвращает вес для компонента стоимости"""
        return self.cost_weights.get_weight(component)

    def update_temperature(self, avg_cost: float, iteration: int) -> float:
        """Адаптивно обновляет температуру"""
        if not self.adaptive_temperature:
            return self.temperature

        # Простая стратегия адаптации температуры
        target_temperature = self.temperature * (1.0 + 0.01 * iteration)
        target_temperature = max(self.min_temperature,
                                 min(self.max_temperature, target_temperature))

        # Плавное обновление
        new_temperature = 0.95 * self.temperature + 0.05 * target_temperature
        self.temperature = new_temperature
        return new_temperature

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"MPPIConfig(\n"
                f"  samples={self.num_samples},\n"
                f"  horizon={self.horizon},\n"
                f"  lambda={self.lambda_},\n"
                f"  control_limit={self.control_limit} N\n"
                f")")


@dataclass
class State:
    """Состояние системы

    Паттерн: Value Object
    """
    x: float = 0.0  # положение тележки (м)
    theta: float = 0.0  # угол маятника (рад)
    x_dot: float = 0.0  # скорость тележки (м/с)
    theta_dot: float = 0.0  # угловая скорость маятника (рад/с)

    def __post_init__(self):
        """Нормализует угол в диапазон [-π, π]"""
        self.normalize_angle()

    def normalize_angle(self):
        """Нормализует угол в диапазон [-π, π]"""
        self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

    def copy(self) -> 'State':
        """Создает копию состояния"""
        return State(self.x, self.theta, self.x_dot, self.theta_dot)

    def to_array(self) -> np.ndarray:
        """Конвертирует в массив NumPy"""
        return np.array([self.x, self.theta, self.x_dot, self.theta_dot],
                        dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Конвертирует в словарь"""
        return {
            'x': self.x,
            'theta': self.theta,
            'x_dot': self.x_dot,
            'theta_dot': self.theta_dot
        }

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'State':
        """Создает из массива NumPy"""
        return cls(x=float(arr[0]), theta=float(arr[1]),
                   x_dot=float(arr[2]), theta_dot=float(arr[3]))

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'State':
        """Создает из словаря"""
        return cls(**data)

    def __str__(self) -> str:
        """Строковое представление"""
        return (f"State(x={self.x:.3f} m, "
                f"θ={math.degrees(self.theta):.1f}°, "
                f"dx={self.x_dot:.3f} m/s, "
                f"dθ={math.degrees(self.theta_dot):.1f}°/s)")


@dataclass
class SimulationConfig:
    """Конфигурация симуляции

    Паттерн: Value Object
    """
    # Параметры симуляции
    duration: float = 10.0  # продолжительность симуляции (с)
    dt: float = 0.02  # шаг времени (с)
    real_time: bool = False  # симуляция в реальном времени

    # Начальные условия
    initial_state: State = field(default_factory=State)
    initial_angle_deg: float = 10.0  # начальный угол в градусах
    initial_position: float = 0.0  # начальное положение тележки

    # Параметры отображения
    visualize: bool = True  # визуализировать симуляцию
    save_animation: bool = False  # сохранять анимацию
    animation_fps: int = 30  # FPS анимации
    plot_trajectory: bool = True  # строить графики траектории

    # Параметры записи
    save_results: bool = True  # сохранять результаты
    results_dir: str = "results"  # директория для результатов
    save_frequency: int = 10  # частота сохранения (каждый N-й шаг)

    # Параметры отладки
    debug: bool = False  # режим отладки
    log_level: str = "INFO"  # уровень логирования
    print_progress: bool = True  # выводить прогресс

    def __post_init__(self):
        """Инициализация после создания"""
        # Устанавливаем начальный угол из градусов
        if hasattr(self, 'initial_angle_deg'):
            self.initial_state.theta = math.radians(self.initial_angle_deg)

        # Устанавливаем начальное положение
        self.initial_state.x = self.initial_position

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        data = asdict(self)
        data['initial_state'] = self.initial_state.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Создает из словаря"""
        # Извлекаем initial_state отдельно
        initial_state_data = data.pop('initial_state', {})
        initial_state = State.from_dict(initial_state_data)

        # Создаем конфигурацию
        config = cls(**data)
        config.initial_state = initial_state
        return config


@dataclass
class ParameterRange:
    """Диапазон параметра для экспериментов

    Паттерн: Value Object
    """
    name: str  # имя параметра
    min_value: float  # минимальное значение
    max_value: float  # максимальное значение
    step: float = None  # шаг (если None, то log scale для некоторых параметров)
    scale: str = "linear"  # шкала (linear, log)
    values: List[float] = None  # конкретные значения (если указаны)

    def get_values(self) -> List[float]:
        """Возвращает список значений параметра"""
        if self.values is not None:
            return self.values

        if self.step is None:
            # Для логарифмической шкалы
            if self.scale == "log":
                return np.logspace(
                    np.log10(self.min_value),
                    np.log10(self.max_value),
                    10
                ).tolist()
            # Для линейной шкалы с автоматическим шагом
            else:
                return np.linspace(self.min_value, self.max_value, 10).tolist()
        else:
            # С заданным шагом
            if self.scale == "log":
                num_steps = int((np.log10(self.max_value) - np.log10(self.min_value))
                                / np.log10(self.step)) + 1
                return np.logspace(
                    np.log10(self.min_value),
                    np.log10(self.max_value),
                    num_steps
                ).tolist()
            else:
                num_steps = int((self.max_value - self.min_value) / self.step) + 1
                return np.linspace(self.min_value, self.max_value, num_steps).tolist()


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента

    Паттерн: Builder (используется для построения сложных экспериментов)
    """
    # Параметры эксперимента
    name: str = "experiment"  # имя эксперимента
    num_runs: int = 5  # количество запусков для каждого параметра
    random_seed: int = 42  # seed для воспроизводимости

    # Параметры для варьирования
    varying_parameters: List[ParameterRange] = field(default_factory=list)

    # Фиксированные параметры
    fixed_system_config: SystemConfig = field(default_factory=SystemConfig)
    fixed_mppi_config: MPPIConfig = field(default_factory=MPPIConfig)

    # Параметры оценки
    metrics: List[str] = field(default_factory=lambda: [
        "success_rate",
        "settling_time",
        "overshoot",
        "control_effort",
        "computation_time"
    ])

    # Параметры сохранения
    save_results: bool = True
    results_dir: str = "experiments"
    generate_report: bool = True

    def add_parameter_range(self, param_range: ParameterRange):
        """Добавляет диапазон параметра"""
        self.varying_parameters.append(param_range)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        data = asdict(self)
        data['fixed_system_config'] = self.fixed_system_config.to_dict()
        data['fixed_mppi_config'] = self.fixed_mppi_config.to_dict()
        data['varying_parameters'] = [
            asdict(p) for p in self.varying_parameters
        ]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Создает из словаря"""
        # Извлекаем вложенные объекты
        system_config_data = data.pop('fixed_system_config', {})
        mppi_config_data = data.pop('fixed_mppi_config', {})
        param_ranges_data = data.pop('varying_parameters', [])

        # Создаем объекты
        system_config = SystemConfig.from_dict(system_config_data)
        mppi_config = MPPIConfig.from_dict(mppi_config_data)
        param_ranges = [ParameterRange(**p) for p in param_ranges_data]

        # Создаем конфигурацию
        config = cls(**data)
        config.fixed_system_config = system_config
        config.fixed_mppi_config = mppi_config
        config.varying_parameters = param_ranges

        return config


class ConfigBuilder:
    """Строитель конфигураций

    Паттерн: Builder - для поэтапного построения сложных конфигураций
    """

    def __init__(self):
        """Инициализирует строитель"""
        self.system_config = SystemConfig()
        self.mppi_config = MPPIConfig()
        self.simulation_config = SimulationConfig()

    def with_system_params(self, cart_mass: float = None, pole_mass: float = None,
                           pole_length: float = None, gravity: float = None,
                           dt: float = None) -> 'ConfigBuilder':
        """Устанавливает параметры системы"""
        if cart_mass is not None:
            self.system_config.cart_mass = cart_mass
        if pole_mass is not None:
            self.system_config.pole_mass = pole_mass
        if pole_length is not None:
            self.system_config.pole_length = pole_length
        if gravity is not None:
            self.system_config.gravity = gravity
        if dt is not None:
            self.system_config.dt = dt
        return self

    def with_mppi_params(self, num_samples: int = None, horizon: int = None,
                         lambda_: float = None, noise_sigma: float = None,
                         control_limit: float = None) -> 'ConfigBuilder':
        """Устанавливает параметры MPPI"""
        if num_samples is not None:
            self.mppi_config.num_samples = num_samples
        if horizon is not None:
            self.mppi_config.horizon = horizon
        if lambda_ is not None:
            self.mppi_config.lambda_ = lambda_
        if noise_sigma is not None:
            self.mppi_config.noise_sigma = noise_sigma
        if control_limit is not None:
            self.mppi_config.control_limit = control_limit
        return self

    def with_cost_weights(self, angle: float = None, angular_velocity: float = None,
                          position: float = None, velocity: float = None,
                          control: float = None) -> 'ConfigBuilder':
        """Устанавливает веса стоимости"""
        if angle is not None:
            self.mppi_config.cost_weights.angle = angle
        if angular_velocity is not None:
            self.mppi_config.cost_weights.angular_velocity = angular_velocity
        if position is not None:
            self.mppi_config.cost_weights.position = position
        if velocity is not None:
            self.mppi_config.cost_weights.velocity = velocity
        if control is not None:
            self.mppi_config.cost_weights.control = control
        return self

    def with_simulation_params(self, duration: float = None,
                               initial_angle_deg: float = None,
                               initial_position: float = None,
                               visualize: bool = None) -> 'ConfigBuilder':
        """Устанавливает параметры симуляции"""
        if duration is not None:
            self.simulation_config.duration = duration
        if initial_angle_deg is not None:
            self.simulation_config.initial_angle_deg = initial_angle_deg
        if initial_position is not None:
            self.simulation_config.initial_position = initial_position
        if visualize is not None:
            self.simulation_config.visualize = visualize
        return self

    def build(self) -> Tuple[SystemConfig, MPPIConfig, SimulationConfig]:
        """Строит конфигурации"""
        return self.system_config, self.mppi_config, self.simulation_config

    def save(self, filename: str):
        """Сохраняет конфигурации в файл"""
        configs = {
            'system_config': self.system_config.to_dict(),
            'mppi_config': self.mppi_config.to_dict(),
            'simulation_config': self.simulation_config.to_dict()
        }

        with open(filename, 'w') as f:
            json.dump(configs, f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'ConfigBuilder':
        """Загружает конфигурации из файла"""
        with open(filename, 'r') as f:
            configs = json.load(f)

        builder = cls()
        builder.system_config = SystemConfig.from_dict(configs['system_config'])
        builder.mppi_config = MPPIConfig.from_dict(configs['mppi_config'])
        builder.simulation_config = SimulationConfig.from_dict(
            configs['simulation_config']
        )

        return builder


# Предустановленные конфигурации

def get_preset_config(name: str) -> Tuple[SystemConfig, MPPIConfig]:
    """Возвращает предустановленную конфигурацию

    Args:
        name: имя предустановки (basic, fast, accurate, robust)

    Returns:
        кортеж (SystemConfig, MPPIConfig)
    """
    presets = {
        'basic': (
            SystemConfig(),
            MPPIConfig(num_samples=500, horizon=20, control_limit=5.0)
        ),
        'fast': (
            SystemConfig(dt=0.01),
            MPPIConfig(num_samples=200, horizon=15, lambda_=0.2, control_limit=8.0)
        ),
        'accurate': (
            SystemConfig(dt=0.005),
            MPPIConfig(num_samples=2000, horizon=40, lambda_=0.05, control_limit=15.0)
        ),
        'robust': (
            SystemConfig(friction_cart=0.2, friction_pole=0.05),
            MPPIConfig(
                num_samples=1000,
                horizon=30,
                lambda_=0.1,
                noise_sigma=2.0,
                control_limit=12.0,
                adaptive_temperature=True
            )
        ),
        'learning': (
            SystemConfig(),
            MPPIConfig(
                num_samples=1000,
                horizon=30,
                lambda_=0.1,
                temperature=2.0,
                adaptive_temperature=True,
                noise_adaptive=True
            )
        )
    }

    if name not in presets:
        raise ValueError(f"Неизвестная предустановка: {name}. "
                         f"Доступные: {list(presets.keys())}")

    return presets[name]


def create_default_configs() -> Tuple[SystemConfig, MPPIConfig, SimulationConfig]:
    """Создает конфигурации по умолчанию"""
    return (
        SystemConfig(),
        MPPIConfig(),
        SimulationConfig()
    )