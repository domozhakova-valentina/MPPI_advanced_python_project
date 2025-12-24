"""
Тесты базового класса MPPI
"""
import numpy as np
import pytest
from mppi.base import MPPIBase
from controller.config import MPPIConfig


class TestMPPI(MPPIBase):
    """Тестовая реализация MPPI для тестирования базового класса"""
    
    def compute_control(self, state: np.ndarray) -> float:
        """Простая реализация для тестирования"""
        return 1.0


class TestMPPIBase:
    """Тесты базового класса MPPI"""
    
    def test_initialization(self, mppi_config):
        """Тест инициализации"""
        mppi = TestMPPI(mppi_config)
        assert mppi.config == mppi_config
        assert len(mppi.u) == mppi_config.T
        assert mppi.u.shape == (mppi_config.T,)
        
    def test_reset(self, mppi_config):
        """Тест сброса состояния"""
        mppi = TestMPPI(mppi_config)
        mppi.u = np.ones(mppi_config.T) * 5.0
        mppi.costs_history = [1.0, 2.0, 3.0]
        
        mppi.reset()
        
        assert np.allclose(mppi.u, np.zeros(mppi_config.T))
        assert len(mppi.costs_history) == 0
        
    def test_dynamics(self, mppi_config, sample_state):
        """Тест динамики маятника"""
        mppi = TestMPPI(mppi_config)
        
        # Тест с нулевой силой
        derivatives = mppi._dynamics(sample_state, 0.0)
        assert derivatives.shape == (4,)
        assert isinstance(derivatives[0], float)  # dx = скорость
        assert isinstance(derivatives[1], float)  # dθ = угловая скорость
        
        # Тест с положительной силой
        derivatives = mppi._dynamics(sample_state, 1.0)
        assert derivatives[2] > 0  # ускорение тележки должно быть положительным
        
        # Тест с отрицательной силой
        derivatives = mppi._dynamics(sample_state, -1.0)
        assert derivatives[2] < 0  # ускорение тележки должно быть отрицательным
        
    def test_cost_function(self, mppi_config):
        """Тест функции стоимости"""
        mppi = TestMPPI(mppi_config)
        
        # Создаем тестовые траектории
        T = mppi_config.T
        state_trajectory = np.zeros((T, 4))
        control_trajectory = np.zeros(T)
        
        # Нулевая траектория должна иметь нулевую стоимость (кроме штрафов за управление)
        cost = mppi._cost_function(state_trajectory, control_trajectory)
        assert cost >= 0
        assert isinstance(cost, float)
        
        # Траектория с отклонением должна иметь большую стоимость
        state_trajectory[:, 1] = 0.5  # угол отклонения
        control_trajectory[:] = 1.0   # управление
        
        cost_with_deviation = mppi._cost_function(state_trajectory, control_trajectory)
        assert cost_with_deviation > cost
        
    def test_compute_control_interface(self, mppi_config, sample_state):
        """Тест интерфейса compute_control"""
        mppi = TestMPPI(mppi_config)
        
        force = mppi.compute_control(sample_state)
        
        assert isinstance(force, float)
        assert force == 1.0  # из нашей тестовой реализации


def test_abstract_methods():
    """Тест, что абстрактный класс нельзя инстанциировать напрямую"""
    from abc import ABCMeta
    
    assert MPPIBase.__abstractmethods__ == frozenset({'compute_control'})
    
    with pytest.raises(TypeError):
        MPPIBase(None)