"""
Единый интерфейс контроллера (Facade Pattern)
"""
import numpy as np
from typing import Dict, Any, Optional
from ..mppi import MPPIBase, MPPINumpy, MPPIJax, MPPICpp
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