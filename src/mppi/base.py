from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import math
from enum import Enum


class CostComponent(Enum):
    """Компоненты функции стоимости"""
    ANGLE = "angle"  # Штраф за угол
    ANGULAR_VELOCITY = "angular_velocity"  # Штраф за угловую скорость
    POSITION = "position"  # Штраф за положение тележки
    VELOCITY = "velocity"  # Штраф за скорость тележки
    CONTROL = "control"  # Штраф за управление


@dataclass
class SystemConfig:
    """Конфигурация системы (маятник + тележка)

    Паттерн: Value Object
    """
    cart_mass: float = 1.0  # M - масса тележки (кг)
    pole_mass: float = 0.1  # m - масса маятника (кг)
    pole_length: float = 1.0  # l - длина маятника (м)
    gravity: float = 9.81  # g - ускорение свободного падения (м/с²)
    dt: float = 0.02  # шаг времени для дискретизации (с)
    friction_cart: float = 0.1  # коэффициент трения тележки
    friction_pole: float = 0.01  # коэффициент трения маятника

    def __post_init__(self):
        """Валидация конфигурации"""
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

    def copy(self) -> 'SystemConfig':
        """Создает копию конфигурации"""
        return SystemConfig(
            cart_mass=self.cart_mass,
            pole_mass=self.pole_mass,
            pole_length=self.pole_length,
            gravity=self.gravity,
            dt=self.dt,
            friction_cart=self.friction_cart,
            friction_pole=self.friction_pole
        )

    def to_dict(self) -> Dict[str, float]:
        """Конвертирует в словарь"""
        return {
            'cart_mass': self.cart_mass,
            'pole_mass': self.pole_mass,
            'pole_length': self.pole_length,
            'gravity': self.gravity,
            'dt': self.dt,
            'friction_cart': self.friction_cart,
            'friction_pole': self.friction_pole
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'SystemConfig':
        """Создает из словаря"""
        return cls(**data)


@dataclass
class MPPIConfig:
    """Конфигурация алгоритма MPPI

    Паттерн: Value Object
    """
    num_samples: int = 1000  # K - количество траекторий
    horizon: int = 30  # T - горизонт планирования
    lambda_: float = 0.1  # λ - параметр для вычисления весов
    noise_sigma: float = 1.0  # σ - стандартное отклонение шума
    control_limit: float = 10.0  # максимальное значение силы (Н)
    temperature: float = 1.0  # температура для softmax

    # Веса компонентов стоимости
    cost_weights: Dict[CostComponent, float] = field(
        default_factory=lambda: {
            CostComponent.ANGLE: 10.0,
            CostComponent.ANGULAR_VELOCITY: 0.1,
            CostComponent.POSITION: 1.0,
            CostComponent.VELOCITY: 0.1,
            CostComponent.CONTROL: 0.01
        }
    )

    # Параметры для расширенных функций стоимости
    use_terminal_cost: bool = True  # использовать терминальную стоимость
    use_state_constraints: bool = False  # использовать ограничения состояния
    state_constraints: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            'x': (-2.0, 2.0),  # ограничение положения тележки
            'theta': (-math.pi / 2, math.pi / 2)  # ограничение угла
        }
    )

    def __post_init__(self):
        """Валидация конфигурации"""
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

    def copy(self) -> 'MPPIConfig':
        """Создает копию конфигурации"""
        return MPPIConfig(
            num_samples=self.num_samples,
            horizon=self.horizon,
            lambda_=self.lambda_,
            noise_sigma=self.noise_sigma,
            control_limit=self.control_limit,
            temperature=self.temperature,
            cost_weights=self.cost_weights.copy(),
            use_terminal_cost=self.use_terminal_cost,
            use_state_constraints=self.use_state_constraints,
            state_constraints=self.state_constraints.copy()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            'num_samples': self.num_samples,
            'horizon': self.horizon,
            'lambda_': self.lambda_,
            'noise_sigma': self.noise_sigma,
            'control_limit': self.control_limit,
            'temperature': self.temperature,
            'cost_weights': {k.value: v for k, v in self.cost_weights.items()},
            'use_terminal_cost': self.use_terminal_cost,
            'use_state_constraints': self.use_state_constraints,
            'state_constraints': self.state_constraints
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MPPIConfig':
        """Создает из словаря"""
        # Конвертируем строки обратно в CostComponent
        if 'cost_weights' in data:
            data['cost_weights'] = {
                CostComponent(k): v for k, v in data['cost_weights'].items()
            }
        return cls(**data)


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
        self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

    def normalize(self) -> 'State':
        """Возвращает нормализованную версию состояния"""
        theta_normalized = ((self.theta + math.pi) % (2 * math.pi)) - math.pi
        return State(self.x, theta_normalized, self.x_dot, self.theta_dot)

    def to_array(self) -> np.ndarray:
        """Конвертирует в массив NumPy"""
        return np.array([self.x, self.theta, self.x_dot, self.theta_dot])

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
        return cls(x=arr[0], theta=arr[1], x_dot=arr[2], theta_dot=arr[3])

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'State':
        """Создает из словаря"""
        return cls(**data)

    def __str__(self) -> str:
        """Строковое представление"""
        return f"State(x={self.x:.3f}, θ={math.degrees(self.theta):.1f}°, " \
               f"dx={self.x_dot:.3f}, dθ={math.degrees(self.theta_dot):.1f}°/s)"


class DynamicsModel(ABC):
    """Абстрактный класс модели динамики системы

    Паттерн: Strategy
    """

    @abstractmethod
    def step(self, state: State, control: float, dt: float) -> State:
        """Вычисляет следующее состояние системы

        Args:
            state: текущее состояние
            control: приложенная сила (Н)
            dt: шаг времени (с)

        Returns:
            следующее состояние
        """
        pass

    @abstractmethod
    def derivatives(self, state: State, control: float) -> Tuple[float, float, float, float]:
        """Вычисляет производные состояния

        Args:
            state: текущее состояние
            control: приложенная сила (Н)

        Returns:
            кортеж производных (dx, dtheta, dx_dot, dtheta_dot)
        """
        pass

    @abstractmethod
    def linearize(self, state: State, control: float) -> Tuple[np.ndarray, np.ndarray]:
        """Линеаризует модель вокруг точки

        Args:
            state: состояние для линеаризации
            control: управление для линеаризации

        Returns:
            кортеж (матрица A, матрица B) для ẋ = Ax + Bu
        """
        pass


class InvertedPendulumModel(DynamicsModel):
    """Модель перевернутого маятника на тележке

    Реализует уравнения движения с учетом трения.

    Паттерн: Strategy (конкретная реализация)
    """

    def __init__(self, config: SystemConfig):
        """Инициализирует модель с заданной конфигурацией

        Args:
            config: конфигурация системы
        """
        self.config = config

    def derivatives(self, state: State, control: float) -> Tuple[float, float, float, float]:
        """Вычисляет производные состояния с учетом трения

        Уравнения движения с трением:
        ẍ = [F - f_c·ẋ + m·sinθ·(l·θ̇² + g·cosθ)] / [M + m·sin²θ]
        θ̈ = [-F·cosθ - f_p·θ̇ - m·l·θ̇²·cosθ·sinθ - (M+m)·g·sinθ] / [l·(M + m·sin²θ)]

        Args:
            state: текущее состояние
            control: приложенная сила (Н)

        Returns:
            кортеж производных (dx, dtheta, dx_dot, dtheta_dot)
        """
        # Извлекаем параметры
        M = self.config.cart_mass
        m = self.config.pole_mass
        l = self.config.pole_length
        g = self.config.gravity
        f_c = self.config.friction_cart  # трение тележки
        f_p = self.config.friction_pole  # трение маятника

        # Извлекаем переменные состояния
        theta = state.theta
        theta_dot = state.theta_dot
        x_dot = state.x_dot
        F = control

        # Вычисляем тригонометрические функции
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        sin2_theta = sin_theta * sin_theta

        # Общий знаменатель для уравнений
        denom = M + m * sin2_theta

        # Ускорение тележки (ẍ) с учетом трения
        x_ddot = (F - f_c * x_dot + m * sin_theta *
                  (l * theta_dot * theta_dot + g * cos_theta)) / denom

        # Угловое ускорение маятника (θ̈) с учетом трения
        theta_ddot = (-F * cos_theta - f_p * theta_dot -
                      m * l * theta_dot * theta_dot * cos_theta * sin_theta -
                      (M + m) * g * sin_theta) / (l * denom)

        return (x_dot, theta_dot, x_ddot, theta_ddot)

    def step(self, state: State, control: float, dt: float) -> State:
        """Вычисляет следующее состояние с помощью метода Рунге-Кутты 4-го порядка

        Args:
            state: текущее состояние
            control: приложенная сила (Н)
            dt: шаг времени (с)

        Returns:
            следующее состояние
        """
        # Ограничиваем управление (опционально, можно делать в контроллере)
        control = max(-self.config.cart_mass * 5,
                      min(self.config.cart_mass * 5, control))

        # Метод Рунге-Кутты 4-го порядка
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

        return next_state.normalize()

    def linearize(self, state: State, control: float) -> Tuple[np.ndarray, np.ndarray]:
        """Линеаризует модель вокруг точки (для расширенных алгоритмов)

        Args:
            state: состояние для линеаризации
            control: управление для линеаризации

        Returns:
            кортеж (матрица A 4x4, матрица B 4x1)
        """
        # Извлекаем параметры
        M = self.config.cart_mass
        m = self.config.pole_mass
        l = self.config.pole_length
        g = self.config.gravity
        f_c = self.config.friction_cart
        f_p = self.config.friction_pole

        # Точка линеаризации (вертикальное положение)
        theta = state.theta
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Вычисляем элементы матриц
        denom = M + m * sin_theta * sin_theta

        # Матрица A (4x4)
        A = np.zeros((4, 4))

        # dx/dx = 0, dx/dtheta = 0, dx/dx_dot = 1, dx/dtheta_dot = 0
        A[0, 2] = 1.0

        # dtheta/dx = 0, dtheta/dtheta = 0, dtheta/dx_dot = 0, dtheta/dtheta_dot = 1
        A[1, 3] = 1.0

        # dx_dot/dx = 0
        # dx_dot/dtheta = сложное выражение
        A[2, 1] = (m * cos_theta * (l * state.theta_dot ** 2 + g * cos_theta) -
                   m * sin_theta * (-g * sin_theta)) / denom - \
                  (2 * m * sin_theta * cos_theta *
                   (control - f_c * state.x_dot + m * sin_theta *
                    (l * state.theta_dot ** 2 + g * cos_theta))) / (denom ** 2)

        # dx_dot/dx_dot = -f_c / denom
        A[2, 2] = -f_c / denom

        # dx_dot/dtheta_dot = 2 * m * l * sin_theta * state.theta_dot / denom
        A[2, 3] = 2 * m * l * sin_theta * state.theta_dot / denom

        # dtheta_dot/dx = 0
        # dtheta_dot/dtheta = сложное выражение
        A[3, 1] = (control * sin_theta - f_p * state.theta_dot * 0 -
                   m * l * state.theta_dot ** 2 *
                   (cos_theta * cos_theta - sin_theta * sin_theta) -
                   (M + m) * g * cos_theta) / (l * denom) - \
                  (2 * m * sin_theta * cos_theta *
                   (-control * cos_theta - f_p * state.theta_dot -
                    m * l * state.theta_dot ** 2 * cos_theta * sin_theta -
                    (M + m) * g * sin_theta)) / (l * denom ** 2)

        # dtheta_dot/dx_dot = 0
        # dtheta_dot/dtheta_dot = -f_p / (l * denom)
        A[3, 3] = -f_p / (l * denom) - \
                  (2 * m * l * state.theta_dot * cos_theta * sin_theta) / denom

        # Матрица B (4x1)
        B = np.zeros((4, 1))

        # dx/du = 0
        # dtheta/du = 0

        # dx_dot/du = 1 / denom
        B[2, 0] = 1.0 / denom

        # dtheta_dot/du = -cos_theta / (l * denom)
        B[3, 0] = -cos_theta / (l * denom)

        return A, B


class MPPIBase(ABC):
    """Абстрактный базовый класс для всех реализаций алгоритма MPPI

    Паттерн: Template Method - определяет структуру алгоритма,
             позволяя подклассам переопределять отдельные шаги
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
        self.model = self._create_model()

        # Номинальная траектория управления
        self.nominal_controls = [0.0] * mppi_config.horizon

        # История для анализа
        self.history = {
            'states': [],
            'controls': [],
            'costs': [],
            'compute_times': []
        }

        # Счетчик итераций
        self.iteration = 0

        # Флаг отладки
        self.debug = False

    @abstractmethod
    def _create_model(self) -> DynamicsModel:
        """Создает модель динамики

        Returns:
            экземпляр модели динамики
        """
        pass

    @abstractmethod
    def _generate_noise(self, num_samples: int, horizon: int) -> np.ndarray:
        """Генерирует случайные возмущения

        Args:
            num_samples: количество траекторий
            horizon: горизонт планирования

        Returns:
            массив шумов формы (num_samples, horizon)
        """
        pass

    @abstractmethod
    def _rollout_trajectories(self, initial_state: State,
                              controls_batch: np.ndarray) -> np.ndarray:
        """Прокручивает траектории для батча управлений

        Args:
            initial_state: начальное состояние
            controls_batch: батч управлений формы (num_samples, horizon)

        Returns:
            массив конечных состояний или стоимостей
        """
        pass

    @abstractmethod
    def _compute_weights(self, costs: np.ndarray) -> np.ndarray:
        """Вычисляет веса для траекторий

        Args:
            costs: массив стоимостей формы (num_samples,)

        Returns:
            массив весов формы (num_samples,)
        """
        pass

    def compute_cost(self, trajectory: List[State], controls: List[float]) -> float:
        """Вычисляет стоимость траектории

        Паттерн: Strategy - функция стоимости может быть изменена

        Args:
            trajectory: список состояний на траектории
            controls: список управлений на траектории

        Returns:
            общая стоимость траектории
        """
        total_cost = 0.0

        for t, (state, control) in enumerate(zip(trajectory, controls)):
            # Базовые компоненты стоимости
            cost = 0.0

            # Штраф за угол
            if CostComponent.ANGLE in self.mppi_config.cost_weights:
                angle_weight = self.mppi_config.cost_weights[CostComponent.ANGLE]
                cost += angle_weight * state.theta ** 2

            # Штраф за угловую скорость
            if CostComponent.ANGULAR_VELOCITY in self.mppi_config.cost_weights:
                ang_vel_weight = self.mppi_config.cost_weights[CostComponent.ANGULAR_VELOCITY]
                cost += ang_vel_weight * state.theta_dot ** 2

            # Штраф за положение
            if CostComponent.POSITION in self.mppi_config.cost_weights:
                pos_weight = self.mppi_config.cost_weights[CostComponent.POSITION]
                cost += pos_weight * state.x ** 2

            # Штраф за скорость
            if CostComponent.VELOCITY in self.mppi_config.cost_weights:
                vel_weight = self.mppi_config.cost_weights[CostComponent.VELOCITY]
                cost += vel_weight * state.x_dot ** 2

            # Штраф за управление
            if CostComponent.CONTROL in self.mppi_config.cost_weights:
                control_weight = self.mppi_config.cost_weights[CostComponent.CONTROL]
                cost += control_weight * control ** 2

            # Ограничения состояния (штрафы за нарушение)
            if self.mppi_config.use_state_constraints:
                constraints = self.mppi_config.state_constraints

                # Проверка положения тележки
                if 'x' in constraints:
                    x_min, x_max = constraints['x']
                    if state.x < x_min:
                        cost += 1000.0 * (x_min - state.x) ** 2
                    elif state.x > x_max:
                        cost += 1000.0 * (state.x - x_max) ** 2

                # Проверка угла маятника
                if 'theta' in constraints:
                    theta_min, theta_max = constraints['theta']
                    if state.theta < theta_min:
                        cost += 1000.0 * (theta_min - state.theta) ** 2
                    elif state.theta > theta_max:
                        cost += 1000.0 * (state.theta - theta_max) ** 2

            total_cost += cost

        # Терминальная стоимость (дополнительный штраф в конце)
        if self.mppi_config.use_terminal_cost and trajectory:
            final_state = trajectory[-1]
            terminal_cost = (
                    50.0 * final_state.theta ** 2 +
                    5.0 * final_state.theta_dot ** 2 +
                    10.0 * final_state.x ** 2 +
                    1.0 * final_state.x_dot ** 2
            )
            total_cost += terminal_cost

        return total_cost

    def compute_control(self, current_state: State) -> float:
        """Основной метод MPPI - вычисляет управляющее воздействие

        Паттерн: Template Method - определяет структуру алгоритма

        Args:
            current_state: текущее состояние системы

        Returns:
            управляющее воздействие (сила)
        """
        import time
        start_time = time.time()

        # 1. Генерируем случайные возмущения
        noise = self._generate_noise(
            self.mppi_config.num_samples,
            self.mppi_config.horizon
        )

        # 2. Создаем управления для каждой траектории
        controls_batch = self.nominal_controls + noise

        # Ограничиваем управления
        controls_batch = np.clip(
            controls_batch,
            -self.mppi_config.control_limit,
            self.mppi_config.control_limit
        )

        # 3. Прокручиваем траектории и вычисляем стоимости
        costs = np.zeros(self.mppi_config.num_samples)

        for i in range(self.mppi_config.num_samples):
            # Прокручиваем траекторию
            trajectory = []
            state = current_state

            for t in range(self.mppi_config.horizon):
                control = controls_batch[i, t]
                state = self.model.step(state, control, self.system_config.dt)
                trajectory.append(state)

            # Вычисляем стоимость
            costs[i] = self.compute_cost(
                trajectory,
                controls_batch[i].tolist()
            )

        # 4. Вычисляем веса
        weights = self._compute_weights(costs)

        # 5. Обновляем номинальную траекторию
        weighted_noise = np.sum(weights[:, np.newaxis] * noise, axis=0)
        self.nominal_controls = self.nominal_controls + weighted_noise

        # Ограничиваем номинальную траекторию
        self.nominal_controls = np.clip(
            self.nominal_controls,
            -self.mppi_config.control_limit,
            self.mppi_config.control_limit
        ).tolist()

        # 6. Извлекаем управление для текущего шага
        control_to_apply = self.nominal_controls[0]

        # 7. Сдвигаем траекторию (shift-and-last)
        self.nominal_controls = self.nominal_controls[1:] + [0.0]

        # 8. Сохраняем историю
        compute_time = time.time() - start_time
        self._update_history(current_state, control_to_apply, compute_time)

        self.iteration += 1

        if self.debug and self.iteration % 100 == 0:
            print(f"Iteration {self.iteration}: "
                  f"control = {control_to_apply:.3f}, "
                  f"time = {compute_time * 1000:.2f}ms")

        return float(control_to_apply)

    def _update_history(self, state: State, control: float, compute_time: float):
        """Обновляет историю работы контроллера

        Args:
            state: текущее состояние
            control: примененное управление
            compute_time: время вычисления
        """
        self.history['states'].append(state)
        self.history['controls'].append(control)
        self.history['compute_times'].append(compute_time)

        # Вычисляем стоимость текущего состояния
        cost = self.compute_cost([state], [control])
        self.history['costs'].append(cost)

    def reset(self):
        """Сбрасывает контроллер в начальное состояние"""
        self.nominal_controls = [0.0] * self.mppi_config.horizon
        self.history = {
            'states': [],
            'controls': [],
            'costs': [],
            'compute_times': []
        }
        self.iteration = 0

    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику работы контроллера

        Returns:
            словарь со статистикой
        """
        if not self.history['compute_times']:
            return {
                'iterations': self.iteration,
                'avg_compute_time': 0.0,
                'total_compute_time': 0.0,
                'avg_cost': 0.0,
                'last_state': None
            }

        compute_times = self.history['compute_times']
        costs = self.history['costs']

        return {
            'iterations': self.iteration,
            'avg_compute_time': sum(compute_times) / len(compute_times),
            'total_compute_time': sum(compute_times),
            'min_compute_time': min(compute_times),
            'max_compute_time': max(compute_times),
            'avg_cost': sum(costs) / len(costs) if costs else 0.0,
            'last_state': self.history['states'][-1] if self.history['states'] else None,
            'last_control': self.history['controls'][-1] if self.history['controls'] else 0.0
        }

    def save_trajectory(self, filename: str):
        """Сохраняет траекторию в файл

        Args:
            filename: имя файла для сохранения
        """
        import json

        data = {
            'system_config': self.system_config.to_dict(),
            'mppi_config': self.mppi_config.to_dict(),
            'trajectory': [
                {
                    'step': i,
                    'state': state.to_dict(),
                    'control': control,
                    'cost': cost,
                    'compute_time': compute_time
                }
                for i, (state, control, cost, compute_time) in enumerate(zip(
                    self.history['states'],
                    self.history['controls'],
                    self.history['costs'],
                    self.history['compute_times']
                ))
            ],
            'stats': self.get_stats()
        }

        # Конвертируем numpy массивы в списки
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        data_serializable = convert_to_serializable(data)

        with open(filename, 'w') as f:
            json.dump(data_serializable, f, indent=2, ensure_ascii=False)

    def set_debug(self, debug: bool):
        """Включает/выключает режим отладки

        Args:
            debug: True для включения отладки
        """
        self.debug = debug

    def __str__(self) -> str:
        """Строковое представление контроллера"""
        stats = self.get_stats()
        return f"MPPIController(iterations={stats['iterations']}, " \
               f"avg_time={stats['avg_compute_time'] * 1000:.1f}ms, " \
               f"avg_cost={stats['avg_cost']:.2f})"


# Фабрика для создания контроллеров
class MPPIFactory:
    """Фабрика для создания контроллеров MPPI

    Паттерн: Factory
    """

    @staticmethod
    def create_numpy(system_config: SystemConfig = None,
                     mppi_config: MPPIConfig = None) -> 'MPPIBase':
        """Создает NumPy реализацию контроллера"""
        from .numpy import MPPIController
        if system_config is None:
            system_config = SystemConfig()
        if mppi_config is None:
            mppi_config = MPPIConfig()
        return MPPIController(system_config, mppi_config)

    @staticmethod
    def create_jax(system_config: SystemConfig = None,
                   mppi_config: MPPIConfig = None,
                   key: Any = None) -> 'MPPIBase':
        """Создает JAX реализацию контроллера"""
        from .jax import MPPIController
        if system_config is None:
            system_config = SystemConfig()
        if mppi_config is None:
            mppi_config = MPPIConfig()
        return MPPIController(system_config, mppi_config, key)

    @staticmethod
    def create_cpp(system_config: SystemConfig = None,
                   mppi_config: MPPIConfig = None) -> 'MPPIBase':
        """Создает C++ реализацию контроллера"""
        from .cpp import MPPIController
        if system_config is None:
            system_config = SystemConfig()
        if mppi_config is None:
            mppi_config = MPPIConfig()
        return MPPIController(system_config, mppi_config)

    @staticmethod
    def create(mode: str, **kwargs) -> 'MPPIBase':
        """Создает контроллер указанного типа

        Args:
            mode: тип реализации ('numpy', 'jax', 'cpp')
            **kwargs: дополнительные аргументы

        Returns:
            экземпляр контроллера

        Raises:
            ValueError: если указан неподдерживаемый тип
        """
        if mode == 'numpy':
            return MPPIFactory.create_numpy(**kwargs)
        elif mode == 'jax':
            return MPPIFactory.create_jax(**kwargs)
        elif mode == 'cpp':
            return MPPIFactory.create_cpp(**kwargs)
        else:
            raise ValueError(f"Неподдерживаемый тип: {mode}")
