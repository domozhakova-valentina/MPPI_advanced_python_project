"""
Тесты NumPy реализации MPPI
"""
import numpy as np
import pytest
from mppi.numpy.mppi_numpy import MPPINumpy


class TestMPPINumpy:
    """Тесты NumPy реализации MPPI"""
    
    def test_initialization(self, pendulum_config, mppi_config):
        """Тест инициализации NumPy реализации"""
        from controller.mppi_controller import CombinedConfig
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPINumpy(combined_config)
        
        assert mppi.config == combined_config
        assert len(mppi.u) == mppi_config.T
        assert isinstance(mppi.u, np.ndarray)
        
    def test_compute_control(self, pendulum_config, mppi_config, sample_state):
        """Тест вычисления управления"""
        from controller.mppi_controller import CombinedConfig
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPINumpy(combined_config)
        
        # Вычисляем управление
        force = mppi.compute_control(sample_state)
        
        # Проверяем результат
        assert isinstance(force, float)
        assert abs(force) < 100  # управление должно быть разумным
        
        # Проверяем, что история стоимостей обновилась
        assert len(mppi.costs_history) == 1
        
    def test_multiple_steps(self, pendulum_config, mppi_config, sample_state):
        """Тест нескольких шагов вычисления"""
        from controller.mppi_controller import CombinedConfig
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPINumpy(combined_config)
        
        forces = []
        for _ in range(5):
            force = mppi.compute_control(sample_state)
            forces.append(force)
        
        # Проверяем, что все управления вычислены
        assert len(forces) == 5
        assert len(mppi.costs_history) == 5
        
        # Проверяем, что стоимости положительные
        for cost in mppi.costs_history:
            assert cost >= 0
            
    def test_reset_clears_history(self, pendulum_config, mppi_config, sample_state):
        """Тест, что reset очищает историю"""
        from controller.mppi_controller import CombinedConfig
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPINumpy(combined_config)
        
        # Выполняем несколько шагов
        for _ in range(3):
            mppi.compute_control(sample_state)
        
        # Проверяем, что история заполнена
        assert len(mppi.costs_history) == 3
        
        # Сбрасываем
        mppi.reset()
        
        # Проверяем, что история очищена
        assert len(mppi.costs_history) == 0
        assert np.allclose(mppi.u, np.zeros(mppi_config.T))
        
    def test_different_parameters(self, pendulum_config):
        """Тест с разными параметрами MPPI"""
        from controller.config import MPPIConfig
        from controller.mppi_controller import CombinedConfig
        
        # Тест с малым количеством траекторий
        mppi_config_small = MPPIConfig(K=10, T=10, lambda_=0.5, sigma=0.5)
        combined_config = CombinedConfig(pendulum_config, mppi_config_small)
        mppi = MPPINumpy(combined_config)
        
        state = np.array([0.0, 0.2, 0.0, 0.0])
        force = mppi.compute_control(state)
        assert isinstance(force, float)
        
        # Тест с большим горизонтом планирования
        mppi_config_large = MPPIConfig(K=500, T=100, lambda_=2.0, sigma=2.0)
        combined_config = CombinedConfig(pendulum_config, mppi_config_large)
        mppi = MPPINumpy(combined_config)
        
        force = mppi.compute_control(state)
        assert isinstance(force, float)