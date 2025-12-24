<<<<<<< HEAD
"""
Реализация MPPI на чистом NumPy
"""
import numpy as np
from ..base import MPPIBase
from typing import List


class MPPINumpy(MPPIBase):
    """MPPI реализация на NumPy"""
    
    def compute_control(self, state: np.ndarray) -> float:
        """
        Вычисление управления с использованием NumPy
        """
        # Генерация случайных возмущений
        epsilon = self.config.sigma * np.random.randn(self.config.K, self.config.T)
        
        # Копирование текущей траектории для всех сэмплов
        u_expanded = self.u + np.zeros((self.config.K, self.config.T))
        
        # Стоимости для каждого сэмпла
        costs = np.zeros(self.config.K)
        
        # Оценка стоимости для каждой траектории
        for k in range(self.config.K):
            # Пробная траектория управления
            u_sample = u_expanded[k] + epsilon[k]
            
            # Симуляция траектории
            state_traj = np.zeros((self.config.T, 4))
            current_state = state.copy()
            
            for t in range(self.config.T):
                state_traj[t] = current_state
                # Интегрирование динамики (метод Эйлера)
                derivatives = self._dynamics(current_state, u_sample[t])
                current_state = current_state + derivatives * self.config.dt
            
            # Вычисление стоимости
            costs[k] = self._cost_function(state_traj, u_sample)
        
        # Вычисление весов
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.config.lambda_)
        weights = weights / np.sum(weights)  # нормализация
        
        # Обновление оптимальной траектории
        self.u = self.u + np.sum(weights[:, np.newaxis] * epsilon, axis=0)
        
        # Сохранение истории стоимостей
        self.costs_history.append(min_cost)
        
        # Возврат первого управления
        return float(self.u[0])
=======
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import time
from abc import ABC, abstractmethod
import json


@dataclass
class SystemConfig:
    """Конфигурация системы (маятник + тележка)

    Паттерн: Value Object
    """
    cart_mass: float = 1.0  # M - масса тележки
    pole_mass: float = 0.1  # m - масса маятника
    pole_length: float = 1.0  # l - длина маятника
    gravity: float = 9.81  # g - ускорение свободного падения
    dt: float = 0.02  # шаг времени для дискретизации

    def to_dict(self) -> Dict[str, float]:
        """Конвертирует конфигурацию в словарь"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SystemConfig':
        """Создает конфигурацию из словаря"""
        return cls(**data)


@dataclass
class MPPIConfig:
    """Конфигурация алгоритма MPPI

    Паттерн: Value Object
    """
    num_samples: int = 1000  # K - количество траекторий
    horizon: int = 30  # T - горизонт планирования
    lambda_: float = 0.1  # λ - параметр для вычисления весов (переименовано, т.к. lambda - ключевое слово)
    noise_sigma: float = 1.0  # σ - стандартное отклонение шума
    control_limit: float = 10.0  # максимальное значение силы

    def __post_init__(self):
        """Валидация конфигурации"""
        if self.num_samples <= 0:
            raise ValueError("num_samples должно быть положительным")
        if self.horizon <= 0:
            raise ValueError("horizon должно быть положительным")
        if self.lambda_ <= 0:
            raise ValueError("lambda_ должно быть положительным")
        if self.noise_sigma <= 0:
            raise ValueError("noise_sigma должно быть положительным")
        if self.control_limit <= 0:
            raise ValueError("control_limit должно быть положительным")

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфигурацию в словарь"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPPIConfig':
        """Создает конфигурацию из словаря"""
        return cls(**data)


@dataclass
class State:
    """Состояние системы

    Паттерн: Value Object
    """
    x: float = 0.0  # положение тележки
    theta: float = 0.0  # угол маятника
    x_dot: float = 0.0  # скорость тележки
    theta_dot: float = 0.0  # угловая скорость маятника

    def __post_init__(self):
        """Нормализует угол в диапазон [-π, π]"""
        self.theta = ((self.theta + np.pi) % (2 * np.pi)) - np.pi

    def to_array(self) -> np.ndarray:
        """Конвертирует состояние в массив NumPy"""
        return np.array([self.x, self.theta, self.x_dot, self.theta_dot], dtype=np.float32)

    def to_dict(self) -> Dict[str, float]:
        """Конвертирует состояние в словарь"""
        return asdict(self)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'State':
        """Создает состояние из массива NumPy"""
        return cls(x=arr[0], theta=arr[1], x_dot=arr[2], theta_dot=arr[3])

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'State':
        """Создает состояние из словаря"""
        return cls(**data)


class DynamicsModel(ABC):
    """Абстрактный класс модели динамики системы

    Паттерн: Strategy
    """

    @abstractmethod
    def step(self, state: State, control: float, dt: float) -> State:
        """Вычисляет следующее состояние системы

        Args:
            state: текущее состояние
            control: приложенная сила
            dt: шаг времени

        Returns:
            следующее состояние
        """
        pass

    @abstractmethod
    def derivatives(self, state: State, control: float) -> np.ndarray:
        """Вычисляет производные состояния

        Args:
            state: текущее состояние
            control: приложенная сила

        Returns:
            производные [dx, dtheta, dx_dot, dtheta_dot]
        """
        pass


class InvertedPendulumModel(DynamicsModel):
    """Модель перевернутого маятника на тележке

    Реализует уравнения движения перевернутого маятника.
    Использует метод Рунге-Кутты 4-го порядка для интегрирования.

    Паттерн: Strategy
    """

    def __init__(self, config: SystemConfig):
        """Инициализирует модель с заданной конфигурацией

        Args:
            config: конфигурация системы
        """
        self.config = config
        self._cache = {}  # Кэш для промежуточных вычислений

    def derivatives(self, state: State, control: float) -> np.ndarray:
        """Вычисляет производные состояния системы

        Используются уравнения движения:
        ẍ = [F + m*sinθ*(l*θ̇² + g*cosθ)] / [M + m*sin²θ]
        θ̈ = [-F*cosθ - m*l*θ̇²*cosθ*sinθ - (M+m)*g*sinθ] / [l*(M + m*sin²θ)]

        Args:
            state: текущее состояние
            control: приложенная сила

        Returns:
            массив производных [ẋ, θ̇, ẍ, θ̈]
        """
        # Извлекаем параметры для удобства
        M = self.config.cart_mass
        m = self.config.pole_mass
        l = self.config.pole_length
        g = self.config.gravity

        # Извлекаем переменные состояния
        theta = state.theta
        theta_dot = state.theta_dot
        F = control

        # Вычисляем тригонометрические функции
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin2_theta = sin_theta ** 2

        # Общий знаменатель для уравнений
        denom = M + m * sin2_theta

        # Ускорение тележки (ẍ)
        x_ddot = (F + m * sin_theta * (l * theta_dot ** 2 + g * cos_theta)) / denom

        # Угловое ускорение маятника (θ̈)
        theta_ddot = (-F * cos_theta -
                      m * l * theta_dot ** 2 * cos_theta * sin_theta -
                      (M + m) * g * sin_theta) / (l * denom)

        # Возвращаем производные
        return np.array([state.x_dot, state.theta_dot, x_ddot, theta_ddot], dtype=np.float32)

    def step(self, state: State, control: float, dt: float) -> State:
        """Вычисляет следующее состояние системы

        Использует метод Рунге-Кутты 4-го порядка (RK4) для точного интегрирования.

        Args:
            state: текущее состояние
            control: приложенная сила
            dt: шаг времени

        Returns:
            следующее состояние
        """
        # Ограничиваем управление
        control = np.clip(control, -self.config.cart_mass * 5, self.config.cart_mass * 5)

        # Метод Рунге-Кутты 4-го порядка (RK4)
        k1 = self.derivatives(state, control)

        state2 = State(
            x=state.x + k1[0] * dt / 2,
            theta=state.theta + k1[1] * dt / 2,
            x_dot=state.x_dot + k1[2] * dt / 2,
            theta_dot=state.theta_dot + k1[3] * dt / 2
        )
        k2 = self.derivatives(state2, control)

        state3 = State(
            x=state.x + k2[0] * dt / 2,
            theta=state.theta + k2[1] * dt / 2,
            x_dot=state.x_dot + k2[2] * dt / 2,
            theta_dot=state.theta_dot + k2[3] * dt / 2
        )
        k3 = self.derivatives(state3, control)

        state4 = State(
            x=state.x + k3[0] * dt,
            theta=state.theta + k3[1] * dt,
            x_dot=state.x_dot + k3[2] * dt,
            theta_dot=state.theta_dot + k3[3] * dt
        )
        k4 = self.derivatives(state4, control)

        # Вычисляем конечное состояние
        dx = (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * dt / 6
        dtheta = (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * dt / 6
        dx_dot = (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * dt / 6
        dtheta_dot = (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]) * dt / 6

        next_state = State(
            x=state.x + dx,
            theta=state.theta + dtheta,
            x_dot=state.x_dot + dx_dot,
            theta_dot=state.theta_dot + dtheta_dot
        )

        return next_state

    def vectorized_step(self, states: np.ndarray, controls: np.ndarray, dt: float) -> np.ndarray:
        """Векторизованная версия step для множества состояний

        Args:
            states: массив состояний формы (N, 4)
            controls: массив управлений формы (N,)
            dt: шаг времени

        Returns:
            массив следующих состояний формы (N, 4)
        """
        N = states.shape[0]
        next_states = np.zeros_like(states)

        for i in range(N):
            state = State.from_array(states[i])
            next_state = self.step(state, controls[i], dt)
            next_states[i] = next_state.to_array()

        return next_states


class MPPIController:
    """Реализация алгоритма MPPI на NumPy

    Алгоритм Model Predictive Path Integral (MPPI) для стохастического
    оптимального управления нелинейными системами.

    Паттерн: Template Method
    """

    def __init__(self, system_config: SystemConfig, mppi_config: MPPIConfig):
        """Инициализирует контроллер MPPI

        Args:
            system_config: конфигурация системы
            mppi_config: конфигурация алгоритма MPPI
        """
        self.system_config = system_config
        self.mppi_config = mppi_config

        # Создаем модель динамики
        self.model = InvertedPendulumModel(system_config)

        # Инициализируем номинальную траекторию управления нулями
        self.nominal_controls = np.zeros(mppi_config.horizon, dtype=np.float32)

        # Инициализируем генератор случайных чисел
        self.rng = np.random.RandomState()

        # Статистика работы
        self.stats = {
            'iteration': 0,
            'avg_compute_time': 0.0,
            'total_compute_time': 0.0,
            'cost_history': [],
            'control_history': []
        }

        print(f"NumPy MPPI Controller initialized:")
        print(f"  Samples (K): {mppi_config.num_samples}")
        print(f"  Horizon (T): {mppi_config.horizon}")
        print(f"  Lambda (λ): {mppi_config.lambda_}")

    def compute_cost(self, trajectory: List[State], controls: List[float]) -> float:
        """Вычисляет стоимость траектории

        Функция стоимости штрафует за:
        1. Отклонение маятника от вертикального положения
        2. Отклонение тележки от центра
        3. Большие управляющие воздействия

        Args:
            trajectory: список состояний на траектории
            controls: список управлений на траектории

        Returns:
            общая стоимость траектории
        """
        total_cost = 0.0

        for t, (state, control) in enumerate(zip(trajectory, controls)):
            # Штраф за угол маятника (цель: θ = 0)
            angle_cost = 10.0 * state.theta ** 2

            # Штраф за угловую скорость
            angular_velocity_cost = 0.1 * state.theta_dot ** 2

            # Штраф за положение тележки (цель: x = 0)
            position_cost = 1.0 * state.x ** 2

            # Штраф за скорость тележки
            velocity_cost = 0.1 * state.x_dot ** 2

            # Штраф за управление (минимизация усилий)
            control_cost = 0.01 * control ** 2

            # Дополнительный штраф за большие отклонения
            if abs(state.theta) > np.pi / 4:  # 45 градусов
                angle_cost *= 2.0

            # Суммируем стоимость для этого шага
            total_cost += (angle_cost + angular_velocity_cost +
                           position_cost + velocity_cost + control_cost)

        return total_cost

    def rollout_trajectory(self, initial_state: State, controls: List[float]) -> List[State]:
        """Прокручивает траекторию на горизонте планирования

        Симулирует поведение системы на T шагов вперед
        при заданной последовательности управлений.

        Args:
            initial_state: начальное состояние
            controls: последовательность управлений

        Returns:
            траектория состояний
        """
        trajectory = []
        current_state = initial_state

        for control in controls:
            # Ограничиваем управление
            clamped_control = np.clip(
                control,
                -self.mppi_config.control_limit,
                self.mppi_config.control_limit
            )

            # Вычисляем следующее состояние
            current_state = self.model.step(
                current_state,
                clamped_control,
                self.system_config.dt
            )

            trajectory.append(current_state)

        return trajectory

    def compute_control(self, current_state: State) -> float:
        """Основной метод MPPI - вычисляет управляющее воздействие

        Алгоритм:
        1. Генерируем K случайных возмущений
        2. Для каждой траектории вычисляем стоимость
        3. Вычисляем веса по формуле exp(-(cost - min_cost)/λ)
        4. Усредняем возмущения с весами
        5. Обновляем номинальную траекторию
        6. Возвращаем первое управление

        Args:
            current_state: текущее состояние системы

        Returns:
            управляющее воздействие (сила)
        """
        start_time = time.time()

        K = self.mppi_config.num_samples
        T = self.mppi_config.horizon
        lambda_ = self.mppi_config.lambda_

        # 1. Генерируем случайные возмущения
        # Форма: (K, T) - K траекторий по T шагов
        noise = self.rng.normal(
            0,
            self.mppi_config.noise_sigma,
            (K, T)
        ).astype(np.float32)

        # 2. Создаем управления для каждой траектории
        # nominal_controls имеет форму (T,), noise - (K, T)
        # Используем broadcast: nominal_controls расширяется до (1, T)
        # затем добавляется noise
        controls = self.nominal_controls[np.newaxis, :] + noise

        # Ограничиваем управления
        controls = np.clip(
            controls,
            -self.mppi_config.control_limit,
            self.mppi_config.control_limit
        )

        # 3. Вычисляем стоимости для всех траекторий
        costs = np.zeros(K, dtype=np.float32)

        # Векторизованное вычисление (по одной траектории за раз)
        for i in range(K):
            # Прокручиваем траекторию
            trajectory = self.rollout_trajectory(
                current_state,
                controls[i].tolist()
            )

            # Вычисляем стоимость
            costs[i] = self.compute_cost(trajectory, controls[i].tolist())

        # 4. Вычисляем веса
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / lambda_)

        # Нормализуем веса
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weights /= weight_sum
        else:
            # Если все веса нулевые, используем равномерное распределение
            weights = np.ones(K, dtype=np.float32) / K

        # 5. Обновляем номинальную траекторию управления
        # Усредняем возмущения с весами
        # noise: (K, T), weights: (K,) -> weighted_noise: (T,)
        weighted_noise = np.sum(weights[:, np.newaxis] * noise, axis=0)

        # Обновляем управление: u_new = u + Σ(w_i * ε_i)
        self.nominal_controls += weighted_noise

        # Ограничиваем номинальную траекторию
        self.nominal_controls = np.clip(
            self.nominal_controls,
            -self.mppi_config.control_limit,
            self.mppi_config.control_limit
        )

        # 6. Сдвигаем траекторию (Shift-and-last strategy)
        control_to_apply = self.nominal_controls[0]

        # Сдвигаем влево
        self.nominal_controls[:-1] = self.nominal_controls[1:]
        self.nominal_controls[-1] = 0.0  # Последний элемент заполняем нулем

        # Обновляем статистику
        compute_time = time.time() - start_time
        self.stats['iteration'] += 1
        self.stats['total_compute_time'] += compute_time
        self.stats['avg_compute_time'] = (
                self.stats['total_compute_time'] / self.stats['iteration']
        )
        self.stats['cost_history'].append(float(min_cost))
        self.stats['control_history'].append(float(control_to_apply))

        return float(control_to_apply)

    def vectorized_compute_control(self, current_state: State) -> float:
        """Векторизованная версия compute_control

        Более эффективная реализация с использованием
        векторизованных операций NumPy.

        Args:
            current_state: текущее состояние системы

        Returns:
            управляющее воздействие (сила)
        """
        start_time = time.time()

        K = self.mppi_config.num_samples
        T = self.mppi_config.horizon
        lambda_ = self.mppi_config.lambda_
        dt = self.system_config.dt

        # 1. Генерируем случайные возмущения
        noise = self.rng.normal(0, self.mppi_config.noise_sigma, (K, T)).astype(np.float32)

        # 2. Создаем управления для каждой траектории
        controls = self.nominal_controls[np.newaxis, :] + noise
        controls = np.clip(controls, -self.mppi_config.control_limit, self.mppi_config.control_limit)

        # 3. Векторизованное вычисление траекторий
        # Создаем массив состояний для всех траекторий
        states = np.tile(current_state.to_array(), (K, 1))
        total_costs = np.zeros(K, dtype=np.float32)

        # Прокручиваем траектории по шагам времени
        for t in range(T):
            # Получаем управления для этого шага
            control_t = controls[:, t]

            # Вычисляем следующее состояние для всех траекторий
            states = self.model.vectorized_step(states, control_t, dt)

            # Вычисляем стоимость для этого шага
            # Преобразуем массив состояний обратно в объекты State для вычисления стоимости
            # (это можно оптимизировать, но оставим для ясности)

            # Штраф за угол (θ²)
            angle_cost = 10.0 * states[:, 1] ** 2

            # Штраф за угловую скорость (θ̇²)
            angular_velocity_cost = 0.1 * states[:, 3] ** 2

            # Штраф за положение (x²)
            position_cost = 1.0 * states[:, 0] ** 2

            # Штраф за скорость (ẋ²)
            velocity_cost = 0.1 * states[:, 2] ** 2

            # Штраф за управление (F²)
            control_cost = 0.01 * control_t ** 2

            # Дополнительный штраф за большие отклонения
            large_angle_penalty = np.where(np.abs(states[:, 1]) > np.pi / 4, 2.0, 1.0)
            angle_cost *= large_angle_penalty

            # Суммируем стоимость
            total_costs += (angle_cost + angular_velocity_cost +
                            position_cost + velocity_cost + control_cost)

        # 4. Вычисляем веса
        min_cost = np.min(total_costs)
        weights = np.exp(-(total_costs - min_cost) / lambda_)

        # Нормализуем веса
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weights /= weight_sum
        else:
            weights = np.ones(K, dtype=np.float32) / K

        # 5. Обновляем номинальную траекторию
        weighted_noise = np.sum(weights[:, np.newaxis] * noise, axis=0)
        self.nominal_controls += weighted_noise
        self.nominal_controls = np.clip(
            self.nominal_controls,
            -self.mppi_config.control_limit,
            self.mppi_config.control_limit
        )

        # 6. Сдвигаем траекторию
        control_to_apply = float(self.nominal_controls[0])
        self.nominal_controls[:-1] = self.nominal_controls[1:]
        self.nominal_controls[-1] = 0.0

        # Обновляем статистику
        compute_time = time.time() - start_time
        self.stats['iteration'] += 1
        self.stats['total_compute_time'] += compute_time
        self.stats['avg_compute_time'] = (
                self.stats['total_compute_time'] / self.stats['iteration']
        )
        self.stats['cost_history'].append(float(min_cost))
        self.stats['control_history'].append(control_to_apply)

        return control_to_apply

    def reset(self):
        """Сбрасывает контроллер в начальное состояние"""
        self.nominal_controls = np.zeros(self.mppi_config.horizon, dtype=np.float32)
        self.rng = np.random.RandomState()

        # Сброс статистики
        self.stats = {
            'iteration': 0,
            'avg_compute_time': 0.0,
            'total_compute_time': 0.0,
            'cost_history': [],
            'control_history': []
        }

        print("MPPI Controller reset")

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы контроллера"""
        return self.stats.copy()

    def save_config(self, filename: str):
        """Сохраняет конфигурацию в файл JSON"""
        config = {
            'system_config': self.system_config.to_dict(),
            'mppi_config': self.mppi_config.to_dict()
        }

        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_config(cls, filename: str) -> 'MPPIController':
        """Загружает конфигурацию из файла JSON"""
        with open(filename, 'r') as f:
            config = json.load(f)

        system_config = SystemConfig.from_dict(config['system_config'])
        mppi_config = MPPIConfig.from_dict(config['mppi_config'])

        return cls(system_config, mppi_config)


# Вспомогательные функции для удобства использования

def create_default_controller() -> MPPIController:
    """Создает контроллер с параметрами по умолчанию"""
    system_config = SystemConfig()
    mppi_config = MPPIConfig()
    return MPPIController(system_config, mppi_config)


def simulate_step(controller: MPPIController, state: State,
                  use_vectorized: bool = False) -> Tuple[float, Dict[str, Any]]:
    """Выполняет один шаг симуляции

    Args:
        controller: контроллер MPPI
        state: текущее состояние
        use_vectorized: использовать векторизованную версию

    Returns:
        кортеж (управление, статистика шага)
    """
    if use_vectorized:
        control = controller.vectorized_compute_control(state)
    else:
        control = controller.compute_control(state)

    stats = {
        'control': control,
        'iteration': controller.stats['iteration'],
        'avg_compute_time': controller.stats['avg_compute_time'],
        'current_cost': controller.stats['cost_history'][-1] if controller.stats['cost_history'] else 0.0
    }

    return control, stats
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
