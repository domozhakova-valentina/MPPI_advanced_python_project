"""
Тесты NumPy реализации MPPI
"""
import numpy as np
import pytest


class TestMPPINumpy:
    """Тесты NumPy реализации MPPI"""
    
    def test_initialization(self):
        """Тест инициализации NumPy реализации"""
        # Создаем mock конфигурацию
        class MockConfig:
            K = 100
            T = 20
            lambda_ = 1.0
            sigma = 1.0
            Q = [1.0, 10.0, 0.1, 0.1]
            R = 0.1
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
            dt = 0.02
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        assert len(mppi.u) == config.T
        assert isinstance(mppi.u, np.ndarray)
        assert len(mppi.costs_history) == 0
    
    def test_compute_control_basic(self):
        """Базовый тест вычисления управления"""
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
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        state = np.array([0.0, 0.1, 0.0, 0.0])
        force = mppi.compute_control(state)
        
        assert isinstance(force, float)
        assert abs(force) < 50
        assert len(mppi.costs_history) == 1
    
    def test_compute_control_with_different_states(self):
        """Тест с разными начальными состояниями"""
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
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        # Небольшое отклонение
        state1 = np.array([0.0, 0.1, 0.0, 0.0])
        force1 = mppi.compute_control(state1)
        
        # Большее отклонение
        mppi.reset()
        state2 = np.array([0.0, 0.5, 0.0, 0.0])
        force2 = mppi.compute_control(state2)
        
        # Силы должны быть разными
        assert force1 != force2
        assert isinstance(force1, float)
        assert isinstance(force2, float)
    
    def test_reset_functionality(self):
        """Тест сброса"""
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
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        state = np.array([0.0, 0.1, 0.0, 0.0])
        
        # Выполняем несколько вычислений
        for _ in range(3):
            mppi.compute_control(state)
        
        assert len(mppi.costs_history) == 3
        
        # Сбрасываем
        mppi.reset()
        
        # Проверяем сброс
        assert len(mppi.costs_history) == 0
        assert np.allclose(mppi.u, np.zeros(config.T))
    
    def test_trajectory_update(self):
        """Тест обновления траектории"""
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
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        initial_u = mppi.u.copy()
        
        state = np.array([0.0, 0.2, 0.0, 0.0])
        force = mppi.compute_control(state)
        
        # Проверяем, что траектория обновилась
        assert not np.allclose(mppi.u, initial_u)
        assert mppi.u[0] == force  # Первое управление должно совпадать
    
    def test_dynamics_inherited(self):
        """Тест наследования динамики"""
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
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        # Проверяем, что метод динамики существует
        assert hasattr(mppi, '_dynamics')
        assert callable(mppi._dynamics)
        
        state = np.array([0.0, 0.1, 0.0, 0.0])
        derivatives = mppi._dynamics(state, 1.0)
        
        assert derivatives.shape == (4,)
        assert isinstance(derivatives[0], float)
    
    def test_cost_function_inherited(self):
        """Тест наследования функции стоимости"""
        class MockConfig:
            K = 10
            T = 3
            lambda_ = 1.0
            sigma = 0.1
            Q = [1.0, 10.0, 0.1, 0.1]
            R = 0.1
            m_cart = 1.0
            m_pole = 0.1
            l = 1.0
            g = 9.81
            dt = 0.02
        
        from src.mppi.numpy.mppi_numpy import MPPINumpy
        
        config = MockConfig()
        mppi = MPPINumpy(config)
        
        # Проверяем, что метод существует
        assert hasattr(mppi, '_cost_function')
        assert callable(mppi._cost_function)
        
        T = config.T
        state_trajectory = np.zeros((T, 4))
        control_trajectory = np.zeros(T)
        
        cost = mppi._cost_function(state_trajectory, control_trajectory)
        
        assert isinstance(cost, float)
        assert cost >= 0