"""
Тесты контроллера MPPI
"""
import numpy as np
import pytest


class TestMPPIController:
    """Тесты контроллера MPPI"""
    
    def test_controller_initialization(self, pendulum_config, mppi_config):
        """Тест инициализации контроллера"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        assert controller.pendulum_config == pendulum_config
        assert controller.mppi_config == mppi_config
        assert controller.implementation_name == 'numpy'
        assert controller.mppi is not None
        
        # Проверяем начальное состояние
        assert controller.state.shape == (4,)
        assert np.allclose(controller.state, [
            pendulum_config.x0,
            pendulum_config.theta0,
            pendulum_config.dx0,
            pendulum_config.dtheta0
        ])
        
    def test_step_execution(self, pendulum_config, mppi_config):
        """Тест выполнения шага контроллера"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Выполняем шаг
        result = controller.step()
        
        # Проверяем результат
        assert isinstance(result, dict)
        assert 'state' in result
        assert 'force' in result
        assert 'cost' in result
        
        # Проверяем типы данных
        assert isinstance(result['state'], np.ndarray)
        assert result['state'].shape == (4,)
        assert isinstance(result['force'], float)
        assert isinstance(result['cost'], float)
        
        # Проверяем, что история обновилась
        assert len(controller.history['states']) == 1
        assert len(controller.history['controls']) == 1
        assert len(controller.history['costs']) == 1
        
    def test_multiple_steps(self, pendulum_config, mppi_config):
        """Тест нескольких шагов контроллера"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Выполняем несколько шагов
        num_steps = 10
        for _ in range(num_steps):
            controller.step()
        
        # Проверяем историю
        assert len(controller.history['states']) == num_steps
        assert len(controller.history['controls']) == num_steps
        assert len(controller.history['costs']) == num_steps
        
        # Проверяем, что состояния имеют правильную размерность
        for state in controller.history['states']:
            assert state.shape == (4,)
            
    def test_reset_functionality(self, pendulum_config, mppi_config):
        """Тест сброса контроллера"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Выполняем несколько шагов
        for _ in range(5):
            controller.step()
        
        # Проверяем, что история заполнена
        assert len(controller.history['states']) == 5
        
        # Сбрасываем
        controller.reset()
        
        # Проверяем, что история очищена
        assert len(controller.history['states']) == 0
        assert len(controller.history['controls']) == 0
        assert len(controller.history['costs']) == 0
        
        # Проверяем, что состояние вернулось к начальному
        assert np.allclose(controller.state, [
            pendulum_config.x0,
            pendulum_config.theta0,
            pendulum_config.dx0,
            pendulum_config.dtheta0
        ])
        
    def test_performance_metrics(self, pendulum_config, mppi_config):
        """Тест вычисления метрик производительности"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Выполняем несколько шагов
        for _ in range(10):
            controller.step()
        
        # Получаем метрики
        metrics = controller.get_performance_metrics()
        
        # Проверяем, что метрики вычислены
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Проверяем ключевые метрики
        expected_keys = ['avg_angle', 'max_angle', 'avg_force', 'total_cost']
        for key in expected_keys:
            if key in metrics:
                assert isinstance(metrics[key], float)
                
    def test_invalid_implementation(self, pendulum_config, mppi_config):
        """Тест с недопустимой реализацией"""
        from controller.mppi_controller import MPPIController
        
        with pytest.raises(ValueError, match="Недопустимая реализация"):
            MPPIController(pendulum_config, mppi_config, implementation='invalid')