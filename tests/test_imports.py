"""
Тесты импортов
"""
import pytest


def test_import_controller_config():
    """Тест импорта конфигураций контроллера"""
    from src.controller.config import PendulumConfig, MPPIConfig
    
    pendulum = PendulumConfig()
    mppi = MPPIConfig()
    
    assert pendulum.m_cart == 1.0
    assert mppi.K == 1000


def test_import_combined_config():
    """Тест импорта CombinedConfig"""
    from src.combined_config import CombinedConfig
    
    class MockPendulumConfig:
        m_cart = 1.0
        m_pole = 0.1
        l = 1.0
        g = 9.81
        dt = 0.02
    
    class MockMPPIConfig:
        K = 100
        T = 20
        lambda_ = 1.0
        sigma = 1.0
        Q = [1.0, 10.0, 0.1, 0.1]
        R = 0.1
    
    pendulum = MockPendulumConfig()
    mppi = MockMPPIConfig()
    
    config = CombinedConfig(pendulum, mppi)
    
    assert config.m_cart == 1.0
    assert config.K == 100


def test_import_metrics():
    """Тест импорта MetricsCalculator"""
    from src.utils.metrics import MetricsCalculator
    
    calculator = MetricsCalculator
    assert calculator is not None


def test_import_numpy_mppi():
    """Тест импорта MPPINumpy"""
    from src.mppi.numpy.mppi_numpy import MPPINumpy
    
    class MockConfig:
        K = 10
        T = 5
        lambda_ = 1.0
        sigma = 0.1
        Q = [1.0, 10.0, 0.1, 0.1]
        R = 0.1
        m_cart = 1.0
        m_pole = 0.1
        l = 1.0
        g = 9.81
        dt = 0.02
    
    config = MockConfig()
    mppi = MPPINumpy(config)
    
    assert mppi is not None