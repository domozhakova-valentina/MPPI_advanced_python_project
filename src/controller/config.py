"""
Классы конфигурации (Factory Pattern)
"""
from dataclasses import dataclass
from typing import List


@dataclass
class PendulumConfig:
    """Конфигурация физической системы"""
    m_cart: float = 1.0      # масса тележки, кг
    m_pole: float = 0.1      # масса маятника, кг
    l: float = 1.0           # длина маятника, м
    g: float = 9.81          # ускорение свободного падения, м/с²
    dt: float = 0.02         # шаг симуляции, с
    
    # Начальные условия
    x0: float = 0.0          # начальное положение тележки, м
    theta0: float = 0.1      # начальный угол, рад
    dx0: float = 0.0         # начальная скорость тележки, м/с
    dtheta0: float = 0.0     # начальная угловая скорость, рад/с
    
    # Ограничения
    max_force: float = 10.0  # максимальная сила, Н
    max_x: float = 2.0       # максимальное отклонение тележки, м


@dataclass
class MPPIConfig:
    """Конфигурация алгоритма MPPI"""
    # Параметры алгоритма
    K: int = 1000            # количество траекторий
    T: int = 50              # горизонт планирования
    lambda_: float = 1.0     # параметр температуры
    sigma: float = 1.0       # стандартное отклонение шума
    
    # Веса функции стоимости
    Q: List[float] = None    # веса для состояния [x, θ, dx, dθ]
    R: float = 0.1           # вес для управления
    
    def __post_init__(self):
        if self.Q is None:
            self.Q = [1.0, 10.0, 0.1, 0.1]  # по умолчанию