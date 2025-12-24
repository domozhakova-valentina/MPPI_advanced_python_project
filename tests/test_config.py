"""
Тесты классов конфигурации
"""
import pytest
from dataclasses import is_dataclass


class TestConfigurations:
    """Тесты конфигураций"""
    
    def test_pendulum_config_dataclass(self):
        """Тест, что PendulumConfig является датаклассом"""
        from controller.config import PendulumConfig
        
        assert is_dataclass(PendulumConfig)
        
    def test_pendulum_config_defaults(self):
        """Тест значений по умолчанию PendulumConfig"""
        from controller.config import PendulumConfig
        
        config = PendulumConfig()
        
        assert config.m_cart == 1.0
        assert config.m_pole == 0.1
        assert config.l == 1.0
        assert config.g == 9.81
        assert config.dt == 0.02
        assert config.x0 == 0.0
        assert config.theta0 == 0.1
        assert config.dx0 == 0.0
        assert config.dtheta0 == 0.0
        
    def test_pendulum_config_custom_values(self):
        """Тест кастомных значений PendulumConfig"""
        from controller.config import PendulumConfig
        
        config = PendulumConfig(
            m_cart=2.0,
            m_pole=0.2,
            l=1.5,
            g=9.8,
            dt=0.01,
            x0=0.5,
            theta0=0.2,
            dx0=0.1,
            dtheta0=0.05
        )
        
        assert config.m_cart == 2.0
        assert config.m_pole == 0.2
        assert config.l == 1.5
        assert config.g == 9.8
        assert config.dt == 0.01
        assert config.x0 == 0.5
        assert config.theta0 == 0.2
        assert config.dx0 == 0.1
        assert config.dtheta0 == 0.05
        
    def test_mppi_config_dataclass(self):
        """Тест, что MPPIConfig является датаклассом"""
        from controller.config import MPPIConfig
        
        assert is_dataclass(MPPIConfig)
        
    def test_mppi_config_defaults(self):
        """Тест значений по умолчанию MPPIConfig"""
        from controller.config import MPPIConfig
        
        config = MPPIConfig()
        
        assert config.K == 1000
        assert config.T == 50
        assert config.lambda_ == 1.0
        assert config.sigma == 1.0
        assert config.R == 0.1
        assert config.Q == [1.0, 10.0, 0.1, 0.1]
        
    def test_mppi_config_custom_values(self):
        """Тест кастомных значений MPPIConfig"""
        from controller.config import MPPIConfig
        
        config = MPPIConfig(
            K=500,
            T=30,
            lambda_=2.0,
            sigma=0.5,
            R=0.2,
            Q=[2.0, 5.0, 0.5, 0.5]
        )
        
        assert config.K == 500
        assert config.T == 30
        assert config.lambda_ == 2.0
        assert config.sigma == 0.5
        assert config.R == 0.2
        assert config.Q == [2.0, 5.0, 0.5, 0.5]
        
    def test_mppi_config_post_init(self):
        """Тест post_init в MPPIConfig"""
        from controller.config import MPPIConfig
        
        # Создаем без указания Q
        config = MPPIConfig(Q=None)
        
        # Проверяем, что Q установлено по умолчанию
        assert config.Q == [1.0, 10.0, 0.1, 0.1]
        
        # Создаем с указанием Q
        custom_q = [0.5, 5.0, 0.2, 0.2]
        config = MPPIConfig(Q=custom_q)
        
        # Проверяем, что Q не перезаписано
        assert config.Q == custom_q


class TestCombinedConfig:
    """Тесты объединенной конфигурации"""
    
    def test_combined_config_creation(self, pendulum_config, mppi_config):
        """Тест создания CombinedConfig"""
        from controller.mppi_controller import CombinedConfig
        
        combined = CombinedConfig(pendulum_config, mppi_config)
        
        # Проверяем, что значения скопированы правильно
        assert combined.m_cart == pendulum_config.m_cart
        assert combined.m_pole == pendulum_config.m_pole
        assert combined.l == pendulum_config.l
        assert combined.g == pendulum_config.g
        assert combined.dt == pendulum_config.dt
        
        assert combined.K == mppi_config.K
        assert combined.T == mppi_config.T
        assert combined.lambda_ == mppi_config.lambda_
        assert combined.sigma == mppi_config.sigma
        assert combined.Q == mppi_config.Q
        assert combined.R == mppi_config.R