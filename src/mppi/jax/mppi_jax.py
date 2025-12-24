import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, random, lax
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
import equinox as eqx
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Callable, NamedTuple
import numpy as np
from functools import partial
import optax
import time
import json
from abc import ABC, abstractmethod

# Классы:
@dataclass(frozen=True)  # Замораживаем для иммутабельности
class SystemConfig:
    """Конфигурация системы (маятник + тележка)

    Паттерн: Immutable Value Object
    """
    cart_mass: float = 1.0  # M - масса тележки
    pole_mass: float = 0.1  # m - масса маятника
    pole_length: float = 1.0  # l - длина маятника
    gravity: float = 9.81  # g - ускорение свободного падения
    dt: float = 0.02  # шаг времени для дискретизации

    def to_jax(self):
        """Конвертирует конфигурацию в JAX-совместимый формат"""
        return {
            'cart_mass': jnp.array(self.cart_mass, dtype=jnp.float64),
            'pole_mass': jnp.array(self.pole_mass, dtype=jnp.float64),
            'pole_length': jnp.array(self.pole_length, dtype=jnp.float64),
            'gravity': jnp.array(self.gravity, dtype=jnp.float64),
            'dt': jnp.array(self.dt, dtype=jnp.float64)
        }

    @classmethod
    def from_jax(cls, jax_dict):
        """Создает конфигурацию из JAX словаря"""
        return cls(
            cart_mass=float(jax_dict['cart_mass']),
            pole_mass=float(jax_dict['pole_mass']),
            pole_length=float(jax_dict['pole_length']),
            gravity=float(jax_dict['gravity']),
            dt=float(jax_dict['dt'])
        )


@dataclass(frozen=True)
class MPPIConfig:
    """Конфигурация алгоритма MPPI

    Паттерн: Immutable Value Object
    """
    num_samples: int = 1000  # K - количество траекторий
    horizon: int = 30  # T - горизонт планирования
    lambda_: float = 0.1  # λ - параметр для вычисления весов
    noise_sigma: float = 1.0  # σ - стандартное отклонение шума
    control_limit: float = 10.0  # максимальное значение силы
    temperature: float = 1.0  # параметр температуры для softmax
    cost_weights: Tuple[float, float, float, float, float] = (10.0, 0.1, 1.0, 0.1, 0.01)  # веса стоимости

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
        if self.temperature <= 0:
            raise ValueError("temperature должно быть положительным")

    def to_jax(self):
        """Конвертирует конфигурацию в JAX-совместимый формат"""
        return {
            'num_samples': self.num_samples,
            'horizon': self.horizon,
            'lambda_': jnp.array(self.lambda_, dtype=jnp.float64),
            'noise_sigma': jnp.array(self.noise_sigma, dtype=jnp.float64),
            'control_limit': jnp.array(self.control_limit, dtype=jnp.float64),
            'temperature': jnp.array(self.temperature, dtype=jnp.float64),
            'cost_weights': jnp.array(self.cost_weights, dtype=jnp.float64)
        }


class State(NamedTuple):
    """Состояние системы

    NamedTuple для иммутабельности и совместимости с JAX
    Паттерн: Immutable Value Object
    """
    x: jnp.ndarray  # положение тележки
    theta: jnp.ndarray  # угол маятника
    x_dot: jnp.ndarray  # скорость тележки
    theta_dot: jnp.ndarray  # угловая скорость маятника

    @classmethod
    def create(cls, x=0.0, theta=0.0, x_dot=0.0, theta_dot=0.0):
        """Создает состояние с JAX массивами"""
        return cls(
            x=jnp.array(x, dtype=jnp.float64),
            theta=jnp.array(theta, dtype=jnp.float64),
            x_dot=jnp.array(x_dot, dtype=jnp.float64),
            theta_dot=jnp.array(theta_dot, dtype=jnp.float64)
        )

    def normalize_angle(self):
        """Нормализует угол в диапазон [-π, π]"""
        theta_normalized = ((self.theta + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return self._replace(theta=theta_normalized)

    def to_array(self) -> jnp.ndarray:
        """Конвертирует состояние в массив JAX"""
        return jnp.array([self.x, self.theta, self.x_dot, self.theta_dot])

    @classmethod
    def from_array(cls, arr: jnp.ndarray) -> 'State':
        """Создает состояние из массива JAX"""
        return cls(x=arr[0], theta=arr[1], x_dot=arr[2], theta_dot=arr[3])


class InvertedPendulumModel:
    """Модель перевернутого маятника на тележке (обертка для JAX функций)

    Паттерн: Facade
    """

    def __init__(self, config: SystemConfig):
        """Инициализирует модель с заданной конфигурацией

        Args:
            config: конфигурация системы
        """
        self.config = config
        self._step_jitted = jit(partial(rk4_step, config))
        self._derivatives_jitted = jit(partial(derivatives, config))
        self._rollout_jitted = jit(partial(rollout_trajectory, config, config.horizon))

    def step(self, state: State, control: float) -> State:
        """Вычисляет следующее состояние системы

        Args:
            state: текущее состояние
            control: приложенная сила

        Returns:
            следующее состояние
        """
        control_jax = jnp.array(control, dtype=jnp.float64)
        return self._step_jitted(state, control_jax)

    def derivatives(self, state: State, control: float) -> jnp.ndarray:
        """Вычисляет производные состояния

        Args:
            state: текущее состояние
            control: приложенная сила

        Returns:
            производные состояния
        """
        control_jax = jnp.array(control, dtype=jnp.float64)
        return self._derivatives_jitted(state, control_jax)

    def rollout(self, initial_state: State, controls: List[float]) -> State:
        """Прокручивает траекторию

        Args:
            initial_state: начальное состояние
            controls: последовательность управлений

        Returns:
            конечное состояние
        """
        controls_jax = jnp.array(controls, dtype=jnp.float64)
        return self._rollout_jitted(initial_state, controls_jax)


class MPPIController:
    """Реализация алгоритма MPPI на JAX

    Ключевые особенности:
    1. Все ключевые функции скомпилированы с помощью JIT
    2. Используется vmap для параллельного вычисления траекторий
    3. Поддержка автоматического дифференцирования для обучения

    Паттерн: Functional Core, Imperative Shell
    """

    def __init__(self, system_config: SystemConfig, mppi_config: MPPIConfig,
                 key: jnp.ndarray = None):
        """Инициализирует контроллер MPPI

        Args:
            system_config: конфигурация системы
            mppi_config: конфигурация алгоритма MPPI
            key: ключ для генератора случайных чисел JAX
        """
        self.system_config = system_config
        self.mppi_config = mppi_config

        # Инициализируем генератор случайных чисел JAX
        if key is None:
            key = random.PRNGKey(int(time.time()))
        self.key = key

        # Инициализируем номинальную траекторию управления
        self.nominal_controls = jnp.zeros(mppi_config.horizon, dtype=jnp.float64)

        # Создаем модель динамики
        self.model = InvertedPendulumModel(system_config)

        # Компилируем ключевые функции
        self._compute_control_jitted = jit(partial(
            self._compute_control_core,
            system_config,
            mppi_config
        ))

        # Статистика работы
        self.stats = {
            'iteration': 0,
            'compute_times': [],
            'costs': [],
            'controls': []
        }

        print(f"JAX MPPI Controller initialized:")
        print(f"  Device: {jax.default_backend()}")
        print(f"  Samples (K): {mppi_config.num_samples}")
        print(f"  Horizon (T): {mppi_config.horizon}")
        print(f"  Lambda (λ): {mppi_config.lambda_}")

    @staticmethod
    def _compute_control_core(system_config: SystemConfig, mppi_config: MPPIConfig,
                              key: jnp.ndarray, nominal_controls: jnp.ndarray,
                              current_state: State) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Ядро алгоритма MPPI (чистая функция для JIT компиляции)

        Args:
            system_config: конфигурация системы
            mppi_config: конфигурация MPPI
            key: ключ генератора случайных чисел
            nominal_controls: номинальная траектория управления
            current_state: текущее состояние

        Returns:
            кортеж (управление, обновленные номинальные управления, ключ)
        """
        # Извлекаем параметры
        K = mppi_config.num_samples
        T = mppi_config.horizon
        lambda_ = mppi_config.lambda_
        noise_sigma = mppi_config.noise_sigma
        control_limit = mppi_config.control_limit

        # 1. Генерируем случайные возмущения
        key, subkey = random.split(key)
        noise = random.normal(subkey, (K, T)) * noise_sigma

        # 2. Создаем управления для каждой траектории
        controls_batch = nominal_controls + noise

        # Ограничиваем управления
        controls_batch = jnp.clip(controls_batch, -control_limit, control_limit)

        # 3. Прокручиваем траектории (векторизовано)
        # Получаем конечные состояния для всех траекторий
        final_states = batch_rollout_trajectory(system_config, T, current_state, controls_batch)

        # 4. Вычисляем стоимости (векторизовано)
        # Для упрощения считаем стоимость только по конечному состоянию
        # (можно расширить до полной траектории)
        def compute_final_cost(state_arr):
            state = State.from_array(state_arr)
            # Используем только конечную стоимость для скорости
            # В полной версии нужно интегрировать по всей траектории
            cost = compute_cost(mppi_config, state, jnp.array(0.0))
            return cost

        # Векторизуем вычисление стоимости
        costs = vmap(compute_final_cost)(final_states)

        # 5. Вычисляем веса (с температурой для численной устойчивости)
        min_cost = jnp.min(costs)
        shifted_costs = costs - min_cost

        # Softmax с температурой
        weights = jax.nn.softmax(-shifted_costs / (lambda_ * mppi_config.temperature))

        # 6. Обновляем номинальную траекторию
        weighted_noise = jnp.sum(weights[:, jnp.newaxis] * noise, axis=0)
        new_nominal_controls = nominal_controls + weighted_noise
        new_nominal_controls = jnp.clip(new_nominal_controls, -control_limit, control_limit)

        # 7. Извлекаем управление для текущего шага
        control_to_apply = new_nominal_controls[0]

        # 8. Сдвигаем траекторию
        shifted_controls = jnp.roll(new_nominal_controls, -1)
        shifted_controls = shifted_controls.at[-1].set(0.0)  # Последний элемент = 0

        return control_to_apply, shifted_controls, key

    def compute_control(self, current_state: State) -> float:
        """Вычисляет управляющее воздействие для текущего состояния

        Args:
            current_state: текущее состояние системы

        Returns:
            управляющее воздействие (сила)
        """
        start_time = time.time()

        # Выполняем скомпилированную версию алгоритма
        control, new_nominal_controls, new_key = self._compute_control_jitted(
            self.key,
            self.nominal_controls,
            current_state
        )

        # Обновляем внутреннее состояние
        self.nominal_controls = new_nominal_controls
        self.key = new_key

        # Обновляем статистику
        compute_time = time.time() - start_time
        self.stats['iteration'] += 1
        self.stats['compute_times'].append(compute_time)
        self.stats['controls'].append(float(control))

        return float(control)

    def compute_control_with_trajectory(self, current_state: State) -> Tuple[float, jnp.ndarray]:
        """Вычисляет управление и возвращает полную траекторию

        Args:
            current_state: текущее состояние

        Returns:
            кортеж (управление, прогнозируемая траектория)
        """
        # Сначала вычисляем управление
        control = self.compute_control(current_state)

        # Затем прокручиваем траекторию с номинальными управлениями
        trajectory = []
        state = current_state

        for i in range(min(10, self.mppi_config.horizon)):  # Первые 10 шагов для визуализации
            control_i = self.nominal_controls[i] if i < len(self.nominal_controls) else 0.0
            state = self.model.step(state, float(control_i))
            trajectory.append(state.to_array())

        return control, jnp.array(trajectory)

    def reset(self, key: jnp.ndarray = None):
        """Сбрасывает контроллер в начальное состояние

        Args:
            key: новый ключ для генератора случайных чисел (опционально)
        """
        self.nominal_controls = jnp.zeros(self.mppi_config.horizon, dtype=jnp.float64)

        if key is not None:
            self.key = key
        else:
            self.key = random.PRNGKey(int(time.time()))

        # Сброс статистики
        self.stats = {
            'iteration': 0,
            'compute_times': [],
            'costs': [],
            'controls': []
        }

        print("JAX MPPI Controller reset")

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы контроллера"""
        stats = self.stats.copy()

        if stats['compute_times']:
            stats['avg_compute_time'] = np.mean(stats['compute_times'])
            stats['min_compute_time'] = np.min(stats['compute_times'])
            stats['max_compute_time'] = np.max(stats['compute_times'])
        else:
            stats['avg_compute_time'] = 0.0
            stats['min_compute_time'] = 0.0
            stats['max_compute_time'] = 0.0

        return stats

    def save_state(self, filename: str):
        """Сохраняет состояние контроллера

        Args:
            filename: имя файла для сохранения
        """
        state = {
            'system_config': self.system_config.__dict__,
            'mppi_config': self.mppi_config.__dict__,
            'nominal_controls': np.array(self.nominal_controls),
            'key': np.array(self.key),
            'stats': self.stats
        }

        # Конвертируем JAX массивы в numpy для сериализации
        def convert_to_serializable(obj):
            if isinstance(obj, jnp.ndarray):
                return np.array(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        state_serializable = convert_to_serializable(state)

        with open(filename, 'wb') as f:
            np.savez(f, **state_serializable)

    @classmethod
    def load_state(cls, filename: str) -> 'MPPIController':
        """Загружает состояние контроллера из файла

        Args:
            filename: имя файла для загрузки

        Returns:
            загруженный контроллер
        """
        with open(filename, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        # Создаем конфигурации
        system_config = SystemConfig(**data['system_config'].item())
        mppi_config = MPPIConfig(**data['mppi_config'].item())

        # Создаем контроллер
        controller = cls(system_config, mppi_config)

        # Восстанавливаем состояние
        controller.nominal_controls = jnp.array(data['nominal_controls'])
        controller.key = jnp.array(data['key'])
        controller.stats = data['stats'].item()

        return controller


# Чистые функции:
@partial(jit, static_argnums=(0,))
def derivatives(config: SystemConfig, state: State, control: jnp.ndarray) -> jnp.ndarray:
    """Вычисляет производные состояния системы (чистая функция)

    Уравнения движения:
    ẍ = [F + m*sinθ*(l*θ̇² + g*cosθ)] / [M + m*sin²θ]
    θ̈ = [-F*cosθ - m*l*θ̇²*cosθ*sinθ - (M+m)*g*sinθ] / [l*(M + m*sin²θ)]

    Args:
        config: конфигурация системы
        state: текущее состояние
        control: приложенная сила

    Returns:
        массив производных [ẋ, θ̇, ẍ, θ̈]
    """
    # Преобразуем конфигурацию в JAX формат
    config_jax = config.to_jax()

    # Извлекаем параметры
    M = config_jax['cart_mass']
    m = config_jax['pole_mass']
    l = config_jax['pole_length']
    g = config_jax['gravity']

    # Извлекаем переменные состояния
    theta = state.theta
    theta_dot = state.theta_dot
    F = control

    # Вычисляем тригонометрические функции
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    sin2_theta = sin_theta ** 2

    # Общий знаменатель
    denom = M + m * sin2_theta

    # Ускорение тележки (ẍ)
    x_ddot = (F + m * sin_theta * (l * theta_dot ** 2 + g * cos_theta)) / denom

    # Угловое ускорение маятника (θ̈)
    theta_ddot = (-F * cos_theta -
                  m * l * theta_dot ** 2 * cos_theta * sin_theta -
                  (M + m) * g * sin_theta) / (l * denom)

    # Возвращаем производные
    return jnp.array([state.x_dot, state.theta_dot, x_ddot, theta_ddot])


@partial(jit, static_argnums=(0,))
def rk4_step(config: SystemConfig, state: State, control: jnp.ndarray) -> State:
    """Один шаг интегрирования методом Рунге-Кутты 4-го порядка

    Args:
        config: конфигурация системы
        state: текущее состояние
        control: приложенная сила

    Returns:
        следующее состояние
    """
    dt = config.to_jax()['dt']

    # Метод Рунге-Кутты 4-го порядка
    k1 = derivatives(config, state, control)

    state2 = State(
        x=state.x + k1[0] * dt / 2,
        theta=state.theta + k1[1] * dt / 2,
        x_dot=state.x_dot + k1[2] * dt / 2,
        theta_dot=state.theta_dot + k1[3] * dt / 2
    )
    k2 = derivatives(config, state2, control)

    state3 = State(
        x=state.x + k2[0] * dt / 2,
        theta=state.theta + k2[1] * dt / 2,
        x_dot=state.x_dot + k2[2] * dt / 2,
        theta_dot=state.theta_dot + k2[3] * dt / 2
    )
    k3 = derivatives(config, state3, control)

    state4 = State(
        x=state.x + k3[0] * dt,
        theta=state.theta + k3[1] * dt,
        x_dot=state.x_dot + k3[2] * dt,
        theta_dot=state.theta_dot + k3[3] * dt
    )
    k4 = derivatives(config, state4, control)

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

    # Нормализуем угол
    return next_state.normalize_angle()


@partial(jit, static_argnums=(0, 1))
def rollout_trajectory(config: SystemConfig, horizon: int,
                       initial_state: State, controls: jnp.ndarray) -> State:
    """Прокручивает траекторию на горизонте планирования (чистая функция)

    Args:
        config: конфигурация системы
        horizon: горизонт планирования
        initial_state: начальное состояние
        controls: последовательность управлений (длина horizon)

    Returns:
        конечное состояние после прокрутки траектории
    """

    def body_fun(i, state):
        control = controls[i]
        return rk4_step(config, state, control)

    # Используем lax.fori_loop для эффективного цикла
    final_state = lax.fori_loop(0, horizon, body_fun, initial_state)
    return final_state


@partial(jit, static_argnums=(0, 1))
def batch_rollout_trajectory(config: SystemConfig, horizon: int,
                             initial_state: State, controls_batch: jnp.ndarray) -> jnp.ndarray:
    """Векторизованная прокрутка траекторий для батча управлений

    Args:
        config: конфигурация системы
        horizon: горизонт планирования
        initial_state: начальное состояние
        controls_batch: батч управлений формы (batch_size, horizon)

    Returns:
        массив конечных состояний формы (batch_size, 4)
    """
    # Используем vmap для векторизации по батчу
    rollout_vmap = vmap(partial(rollout_trajectory, config, horizon),
                        in_axes=(None, 0))

    # Создаем батч начальных состояний
    initial_states_batch = tree_map(
        lambda x: jnp.tile(x, (controls_batch.shape[0], 1)),
        initial_state.to_array()
    )

    # Конвертируем батч начальных состояний в State NamedTuples
    def create_state_batch(arr):
        return State.from_array(arr)

    initial_states = vmap(create_state_batch)(initial_states_batch)

    # Прокручиваем траектории
    final_states = rollout_vmap(initial_states, controls_batch)

    # Конвертируем в массив
    return vmap(lambda s: s.to_array())(final_states)


@partial(jit, static_argnums=(0,))
def compute_cost(config: MPPIConfig, state: State, control: jnp.ndarray) -> jnp.ndarray:
    """Вычисляет стоимость для одного состояния и управления (чистая функция)

    Args:
        config: конфигурация MPPI
        state: состояние
        control: управление

    Returns:
        стоимость
    """
    config_jax = config.to_jax()
    weights = config_jax['cost_weights']

    # Штраф за угол маятника (цель: θ = 0)
    angle_cost = weights[0] * state.theta ** 2

    # Штраф за угловую скорость
    angular_velocity_cost = weights[1] * state.theta_dot ** 2

    # Штраф за положение тележки (цель: x = 0)
    position_cost = weights[2] * state.x ** 2

    # Штраф за скорость тележки
    velocity_cost = weights[3] * state.x_dot ** 2

    # Штраф за управление (минимизация усилий)
    control_cost = weights[4] * control ** 2

    # Дополнительный штраф за большие отклонения (непрерывная функция)
    large_angle_penalty = 1.0 + jnp.exp(jnp.abs(state.theta) - jnp.pi / 4)
    angle_cost = angle_cost * large_angle_penalty

    # Итоговая стоимость
    total_cost = angle_cost + angular_velocity_cost + position_cost + velocity_cost + control_cost

    return total_cost


@partial(jit, static_argnums=(0, 1))
def compute_trajectory_cost(config: MPPIConfig, horizon: int,
                            states_sequence: jnp.ndarray, controls_sequence: jnp.ndarray) -> jnp.ndarray:
    """Вычисляет общую стоимость для траектории

    Args:
        config: конфигурация MPPI
        horizon: горизонт планирования
        states_sequence: последовательность состояний (horizon, 4)
        controls_sequence: последовательность управлений (horizon,)

    Returns:
        общая стоимость траектории
    """
    # Векторизуем compute_cost по времени
    cost_per_step = vmap(partial(compute_cost, config))(
        vmap(State.from_array)(states_sequence),
        controls_sequence
    )

    # Суммируем стоимость по всем шагам
    total_cost = jnp.sum(cost_per_step)

    return total_cost

# Функции обучения:
def create_trainable_controller(system_config: SystemConfig, mppi_config: MPPIConfig,
                                trainable_params: List[str] = None):
    """Создает контроллер с обучаемыми параметрами

    Args:
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        trainable_params: список параметров для обучения

    Returns:
        кортеж (функция потерь, функция обновления, начальные параметры)
    """
    if trainable_params is None:
        trainable_params = ['lambda_', 'noise_sigma', 'cost_weights']

    # Начальные параметры
    init_params = {
        'lambda_': mppi_config.lambda_,
        'noise_sigma': mppi_config.noise_sigma,
        'cost_weights': jnp.array(mppi_config.cost_weights)
    }

    # Функция потерь (ожидаемая стоимость)
    @jit
    def loss_fn(params, key, initial_state, num_trajectories=100, horizon=50):
        """Вычисляет ожидаемую стоимость для заданных параметров

        Args:
            params: обучаемые параметры
            key: ключ генератора случайных чисел
            initial_state: начальное состояние
            num_trajectories: количество траекторий для оценки
            horizon: горизонт планирования

        Returns:
            средняя стоимость
        """
        # Создаем временную конфигурацию с текущими параметрами
        temp_config = MPPIConfig(
            num_samples=mppi_config.num_samples,
            horizon=horizon,
            lambda_=float(params['lambda_']),
            noise_sigma=float(params['noise_sigma']),
            control_limit=mppi_config.control_limit,
            temperature=mppi_config.temperature,
            cost_weights=tuple(params['cost_weights'])
        )

        # Генерируем случайные управления
        key, subkey = random.split(key)
        controls = random.normal(subkey, (num_trajectories, horizon)) * params['noise_sigma']

        # Прокручиваем траектории
        final_states = batch_rollout_trajectory(system_config, horizon, initial_state, controls)

        # Вычисляем стоимость для каждой траектории
        def compute_trajectory_cost_flat(control_seq, final_state_arr):
            # Упрощенная версия: учитываем только конечное состояние
            state = State.from_array(final_state_arr)
            cost = jnp.sum(control_seq ** 2) * 0.01  # Штраф за управление
            cost += compute_cost(temp_config, state, jnp.array(0.0))
            return cost

        costs = vmap(compute_trajectory_cost_flat)(controls, final_states)

        # Средняя стоимость
        avg_cost = jnp.mean(costs)

        return avg_cost

    # Градиент функции потерь
    grad_loss = jit(grad(loss_fn))

    return loss_fn, grad_loss, init_params


def train_controller(system_config: SystemConfig, mppi_config: MPPIConfig,
                     initial_state: State, num_iterations: int = 1000,
                     learning_rate: float = 0.01):
    """Обучает параметры контроллера с помощью градиентного спуска

    Args:
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        initial_state: начальное состояние для обучения
        num_iterations: количество итераций обучения
        learning_rate: скорость обучения

    Returns:
        обученные параметры
    """
    print("Начинаем обучение контроллера...")

    # Создаем обучаемый контроллер
    loss_fn, grad_fn, params = create_trainable_controller(system_config, mppi_config)

    # Инициализируем оптимизатор
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Ключ для генератора случайных чисел
    key = random.PRNGKey(42)

    # История обучения
    history = []

    for i in range(num_iterations):
        # Обновляем ключ
        key, subkey = random.split(key)

        # Вычисляем градиент
        grads = grad_fn(params, subkey, initial_state)

        # Применяем обновление
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Вычисляем потери
        if i % 100 == 0:
            loss = loss_fn(params, subkey, initial_state)
            history.append((i, float(loss)))
            print(f"Iteration {i}: loss = {float(loss):.4f}")

    print("Обучение завершено!")

    return params, history


def optimize_parameters(system_config: SystemConfig, mppi_config: MPPIConfig,
                        initial_states: List[State], num_iterations: int = 500):
    """Оптимизирует параметры контроллера для набора начальных состояний

    Args:
        system_config: конфигурация системы
        mppi_config: конфигурация MPPI
        initial_states: список начальных состояний
        num_iterations: количество итераций оптимизации

    Returns:
        оптимальные параметры
    """
    print("Оптимизация параметров контроллера...")

    # Преобразуем начальные состояния в массив
    states_array = jnp.array([s.to_array() for s in initial_states])

    # Целевая функция: средняя стоимость по всем начальным состояниям
    @jit
    def objective(params):
        total_cost = 0.0

        # Создаем временную конфигурацию
        temp_config = MPPIConfig(
            num_samples=mppi_config.num_samples,
            horizon=mppi_config.horizon,
            lambda_=float(params['lambda_']),
            noise_sigma=float(params['noise_sigma']),
            control_limit=mppi_config.control_limit,
            temperature=mppi_config.temperature,
            cost_weights=tuple(params['cost_weights'])
        )

        # Вычисляем стоимость для каждого начального состояния
        def compute_for_state(state_arr):
            state = State.from_array(state_arr)

            # Создаем простой контроллер с текущими параметрами
            key = random.PRNGKey(0)
            controller = MPPIController(system_config, temp_config, key)

            # Симулируем несколько шагов
            total_state_cost = 0.0
            current_state = state

            for _ in range(20):  # 20 шагов симуляции
                control = controller.compute_control(current_state)
                current_state = controller.model.step(current_state, float(control))

                # Стоимость текущего состояния
                state_cost = compute_cost(temp_config, current_state, jnp.array(control))
                total_state_cost += state_cost

            return total_state_cost

        # Векторизуем по состояниям
        costs = vmap(compute_for_state)(states_array)

        return jnp.mean(costs)

    # Градиент целевой функции
    grad_objective = jit(grad(objective))

    # Начальные параметры
    params = {
        'lambda_': mppi_config.lambda_,
        'noise_sigma': mppi_config.noise_sigma,
        'cost_weights': jnp.array(mppi_config.cost_weights)
    }

    # Простой градиентный спуск
    learning_rate = 0.01
    history = []

    for i in range(num_iterations):
        grad_params = grad_objective(params)

        # Обновляем параметры
        params = tree_map(
            lambda p, g: p - learning_rate * g,
            params,
            grad_params
        )

        if i % 50 == 0:
            cost = objective(params)
            history.append((i, float(cost)))
            print(f"Iteration {i}: cost = {float(cost):.4f}")

    print("Оптимизация завершена!")

    return params, history

# Вспомогательные функции:
def create_default_controller() -> MPPIController:
    """Создает контроллер с параметрами по умолчанию"""
    system_config = SystemConfig()
    mppi_config = MPPIConfig()
    return MPPIController(system_config, mppi_config)


def simulate_step(controller: MPPIController, state: State) -> Tuple[float, Dict[str, Any]]:
    """Выполняет один шаг симуляции

    Args:
        controller: контроллер MPPI
        state: текущее состояние

    Returns:
        кортеж (управление, статистика шага)
    """
    control = controller.compute_control(state)

    stats = controller.get_stats()
    step_stats = {
        'control': control,
        'iteration': stats['iteration'],
        'avg_compute_time': stats.get('avg_compute_time', 0.0),
        'last_compute_time': stats['compute_times'][-1] if stats['compute_times'] else 0.0
    }

    return control, step_stats
