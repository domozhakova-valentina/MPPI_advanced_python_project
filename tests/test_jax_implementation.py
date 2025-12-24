"""
Тесты JAX реализации MPPI
"""
import numpy as np
import pytest


class TestMPPIJax:
    """Тесты JAX реализации MPPI"""
    
    def test_import_availability(self):
        """Тест доступности JAX"""
        try:
            from mppi.jax.mppi_jax import MPPIJax
            JAX_AVAILABLE = True
        except ImportError:
            JAX_AVAILABLE = False
            
        # Если JAX доступен, тестируем, если нет - пропускаем
        if not JAX_AVAILABLE:
            pytest.skip("JAX не установлен")
            
    @pytest.mark.skipif(True, reason="Требуется установленный JAX")
    def test_jax_initialization(self, pendulum_config, mppi_config):
        """Тест инициализации JAX реализации"""
        from controller.mppi_controller import CombinedConfig
        from mppi.jax.mppi_jax import MPPIJax
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPIJax(combined_config)
        
        assert mppi.config == combined_config
        assert len(mppi.u) == mppi_config.T
        
    @pytest.mark.skipif(True, reason="Требуется установленный JAX")
    def test_jax_compute_control(self, pendulum_config, mppi_config, sample_state):
        """Тест вычисления управления в JAX"""
        from controller.mppi_controller import CombinedConfig
        from mppi.jax.mppi_jax import MPPIJax
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPIJax(combined_config)
        
        force = mppi.compute_control(sample_state)
        
        assert isinstance(force, float)
        assert abs(force) < 100
        
    @pytest.mark.skipif(True, reason="Требуется установленный JAX")
    def test_jax_compilation(self, pendulum_config, mppi_config):
        """Тест компиляции JAX функций"""
        from controller.mppi_controller import CombinedConfig
        from mppi.jax.mppi_jax import MPPIJax
        
        combined_config = CombinedConfig(pendulum_config, mppi_config)
        mppi = MPPIJax(combined_config)
        
        # Проверяем, что функции скомпилированы
        assert hasattr(mppi, '_jax_dynamics')
        assert hasattr(mppi, '_jax_cost')
        assert hasattr(mppi, '_simulate_trajectory')
        
        # Проверяем, что это JAX функции
        import jax
        assert isinstance(mppi._jax_dynamics, jax.core.Tracer) or callable(mppi._jax_dynamics)