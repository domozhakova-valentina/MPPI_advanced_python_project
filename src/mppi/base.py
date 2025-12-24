"""
Базовый класс MPPI (Strategy Pattern)
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List


class MPPIBase(ABC):
    """Абстрактный класс для всех реализаций MPPI"""
    
    def __init__(self, config: 'MPPIConfig'):
        self.config = config
        self.reset()
    
    def reset(self):
        """Сброс состояния контроллера"""
        self.u = np.zeros(self.config.T)  # текущая траектория управления
        self.costs_history = []
    
    @abstractmethod
    def compute_control(self, state: np.ndarray) -> float:
        """
        Вычислить управляющее воздействие для текущего состояния
        
        Args:
            state: [x, θ, dx, dθ] - текущее состояние системы
            
        Returns:
            Управляющая сила F
        """
        pass
    
    def _dynamics(self, state: np.ndarray, F: float) -> np.ndarray:
        """
        Динамика маятника (общая для всех реализаций)
        
        Args:
            state: [x, θ, dx, dθ]
            F: прикладываемая сила
            
        Returns:
            Производные состояния [dx, dθ, ddx, ddθ]
        """
        x, theta, dx, dtheta = state
        M = self.config.m_cart
        m = self.config.m_pole
        l = self.config.l
        g = self.config.g
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        
        # Уравнения динамики
        denom = M + m * sin_theta**2
        
        ddx = (F + m * sin_theta * (l * dtheta**2 + g * cos_theta)) / denom
        ddtheta = (-F * cos_theta - m * l * dtheta**2 * cos_theta * sin_theta - 
                  (M + m) * g * sin_theta) / (l * denom)
        
        return np.array([dx, dtheta, ddx, ddtheta])
    
    def _cost_function(self, state_trajectory: np.ndarray, 
                      control_trajectory: np.ndarray) -> float:
        """
        Функция стоимости
        
        Args:
            state_trajectory: матрица состояний размером (T, 4)
            control_trajectory: вектор управлений размером T
            
        Returns:
            Общая стоимость траектории
        """
        cost = 0.0
        for t in range(self.config.T):
            x, theta, dx, dtheta = state_trajectory[t]
            
            # Штраф за отклонение маятника
            cost += self.config.Q[0] * x**2
            cost += self.config.Q[1] * theta**2
            cost += self.config.Q[2] * dx**2
            cost += self.config.Q[3] * dtheta**2
            
            # Штраф за большие управления
            cost += self.config.R * control_trajectory[t]**2
        
        return cost
