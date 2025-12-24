"""
Конфигурация pytest для тестов MPPI
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к src
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

@pytest.fixture
def pendulum_config():
    """Фикстура конфигурации маятника"""
    from controller.config import PendulumConfig
    return PendulumConfig(
        m_cart=1.0,
        m_pole=0.1,
        l=1.0,
        g=9.81,
        dt=0.02,
        x0=0.0,
        theta0=0.1,
        dx0=0.0,
        dtheta0=0.0
    )

@pytest.fixture
def mppi_config():
    """Фикстура конфигурации MPPI"""
    from controller.config import MPPIConfig
    return MPPIConfig(
        K=100,
        T=20,
        lambda_=1.0,
        sigma=1.0,
        R=0.1,
        Q=[1.0, 10.0, 0.1, 0.1]
    )

@pytest.fixture
def sample_state():
    """Фикстура примерного состояния"""
    return np.array([0.0, 0.1, 0.0, 0.0])  # [x, θ, dx, dθ]

@pytest.fixture
def mock_history():
    """Фикстура истории симуляции"""
    return {
        'states': [np.array([0.0, 0.1, 0.0, 0.0]) for _ in range(10)],
        'controls': [0.1 * np.random.randn() for _ in range(10)],
        'costs': [0.5 + 0.1 * i for i in range(10)]
    }