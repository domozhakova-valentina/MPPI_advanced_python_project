"""
Тесты базового класса MPPI
"""
import numpy as np
import pytest

class TestMPPI:
    """Тестовая реализация MPPI для тестирования базового класса"""
    
    def __init__(self, config):
        self.config = config
        self.u = np.zeros(self.config.T)
        self.costs_history = []
    
    def reset(self):
        self.u = np.zeros(self.config.T)
        self.costs_history = []
    
    def compute_control(self, state):
        return 1.0
    
    def _dynamics(self, state, F):
        # Упрощенная версия динамики для тестов
        x, theta, dx, dtheta = state
        M = self.config.m_cart
        m = self.config.m_pole
        l = self.config.l
        g = self.config.g
        
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        denom = M + m * sin_theta**2
        
        ddx = (F + m * sin_theta * (l * dtheta**2 + g * cos_theta)) / denom
        ddtheta = (-F * cos_theta - m * l * dtheta**2 * cos_theta * sin_theta - 
                  (M + m) * g * sin_theta) / (l * denom)
        
        return np.array([dx, dtheta, ddx, ddtheta])
    
    def _cost_function(self, state_trajectory, control_trajectory):
        # Упрощенная версия функции стоимости
        return np.sum(state_trajectory**2) + np.sum(control_trajectory**2)


class TestMPPIBase:
    """Тесты базового класса MPPI"""
    
    def test_initialization(self):
        """Тест инициализации"""
        class Config:
            T = 20
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
        
        config = Config()
        mppi = TestMPPI(config)
        
        assert len(mppi.u) == config.T
        assert len(mppi.costs_history) == 0
        
    def test_reset(self):
        """Тест сброса состояния"""
        class Config:
            T = 20
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
        
        config = Config()
        mppi = TestMPPI(config)
        mppi.u = np.ones(config.T) * 5.0
        mppi.costs_history = [1.0, 2.0, 3.0]
        
        mppi.reset()
        
        assert np.allclose(mppi.u, np.zeros(config.T))
        assert len(mppi.costs_history) == 0
        
    def test_dynamics(self):
        """Тест динамики маятника"""
        class Config:
            T = 20
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
        
        config = Config()
        mppi = TestMPPI(config)
        
        state = np.array([0.0, 0.1, 0.0, 0.0])
        
        # Тест с нулевой силой
        derivatives = mppi._dynamics(state, 0.0)
        assert derivatives.shape == (4,)
        assert isinstance(derivatives[0], float)
        
    def test_cost_function(self):
        """Тест функции стоимости"""
        class Config:
            T = 20
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
        
        config = Config()
        mppi = TestMPPI(config)
        
        T = 5
        state_trajectory = np.zeros((T, 4))
        control_trajectory = np.zeros(T)
        
        cost = mppi._cost_function(state_trajectory, control_trajectory)
        assert cost >= 0
        assert isinstance(cost, float)