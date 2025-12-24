"""
Конфигурация pytest для тестов MPPI
"""
import pytest
import numpy as np
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Тестовые классы конфигураций
class MockPendulumConfig:
    m_cart = 1.0
    m_pole = 0.1
    l = 1.0
    g = 9.81
    dt = 0.02
    x0 = 0.0
    theta0 = 0.1
    dx0 = 0.0
    dtheta0 = 0.0
    max_force = 10.0
    max_x = 2.0

class MockMPPIConfig:
    K = 100
    T = 20
    lambda_ = 1.0
    sigma = 1.0
    Q = [1.0, 10.0, 0.1, 0.1]
    R = 0.1

class MockCombinedConfig:
    def __init__(self):
        self.m_cart = 1.0
        self.m_pole = 0.1
        self.l = 1.0
        self.g = 9.81
        self.dt = 0.02
        self.K = 100
        self.T = 20
        self.lambda_ = 1.0
        self.sigma = 1.0
        self.Q = [1.0, 10.0, 0.1, 0.1]
        self.R = 0.1

@pytest.fixture
def pendulum_config():
    """Фикстура конфигурации маятника"""
    return MockPendulumConfig()

@pytest.fixture
def mppi_config():
    """Фикстура конфигурации MPPI"""
    return MockMPPIConfig()

@pytest.fixture
def combined_config():
    """Фикстура объединенной конфигурации"""
    return MockCombinedConfig()

@pytest.fixture
def sample_state():
    """Фикстура примера состояния"""
    return np.array([0.0, 0.1, 0.0, 0.0])  # [x, θ, dx, dθ]

@pytest.fixture
def mock_history():
    """Фикстура истории симуляции"""
    return {
        'states': [np.array([0.0, 0.1, 0.0, 0.0]) for _ in range(10)],
        'controls': [0.1 * np.random.randn() for _ in range(10)],
        'costs': [0.5 + 0.1 * i for i in range(10)]
    }