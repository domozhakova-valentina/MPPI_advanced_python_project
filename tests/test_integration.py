"""
Интеграционные тесты системы
"""
import numpy as np


class TestIntegration:
    """Интеграционные тесты системы MPPI"""
    
    def test_full_simulation_flow(self, pendulum_config, mppi_config):
        """Тест полного потока симуляции"""
        from controller.mppi_controller import MPPIController
        from utils.results_collector import ResultsCollector
        
        # Создаем контроллер
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Создаем сборщик результатов
        collector = ResultsCollector()
        
        # Запускаем симуляцию
        result = collector.run_simulation(
            controller=controller,
            num_steps=20,
            experiment_name='integration_test'
        )
        
        # Проверяем, что симуляция прошла успешно
        assert result is not None
        assert 'history' in result
        assert 'metrics' in result
        
        # Проверяем историю
        history = result['history']
        assert len(history['states']) == 20
        assert len(history['controls']) == 20
        assert len(history['costs']) == 20
        
        # Проверяем, что все управления вычислены
        for control in history['controls']:
            assert isinstance(control, float)
            
        # Проверяем, что все состояния имеют правильную размерность
        for state in history['states']:
            assert state.shape == (4,)
            assert not np.any(np.isnan(state))  # нет NaN значений
            
    def test_controller_reset_integration(self, pendulum_config, mppi_config):
        """Интеграционный тест сброса контроллера"""
        from controller.mppi_controller import MPPIController
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Запускаем симуляцию
        for _ in range(10):
            controller.step()
        
        # Проверяем, что история заполнена
        assert len(controller.history['states']) == 10
        
        # Сбрасываем
        controller.reset()
        
        # Запускаем снова
        for _ in range(5):
            controller.step()
        
        # Проверяем, что история содержит только новые данные
        assert len(controller.history['states']) == 5
        
    def test_metrics_calculation_integration(self, pendulum_config, mppi_config):
        """Интеграционный тест вычисления метрик"""
        from controller.mppi_controller import MPPIController
        from utils.metrics import MetricsCalculator
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Запускаем симуляцию
        for _ in range(15):
            controller.step()
        
        # Вычисляем метрики
        metrics = controller.get_performance_metrics()
        comprehensive_metrics = MetricsCalculator.calculate_comprehensive_metrics(controller.history)
        
        # Проверяем, что метрики вычислены
        assert len(metrics) > 0
        assert len(comprehensive_metrics) > 0
        
        # Проверяем согласованность
        if 'avg_angle' in metrics and 'angle_rmse' in comprehensive_metrics:
            # Средний угол должен быть меньше RMSE
            assert metrics['avg_angle'] <= comprehensive_metrics['angle_rmse'] + 0.1
            
    def test_different_configurations_integration(self):
        """Интеграционный тест с разными конфигурациями"""
        from controller.config import PendulumConfig, MPPIConfig
        from controller.mppi_controller import MPPIController
        
        # Легкий маятник
        config_light = PendulumConfig(m_pole=0.05, l=0.5)
        mppi_config = MPPIConfig(K=200, T=30)
        controller = MPPIController(config_light, mppi_config, implementation='numpy')
        
        for _ in range(10):
            result = controller.step()
            assert isinstance(result['force'], float)
        
        # Тяжелый маятник
        config_heavy = PendulumConfig(m_pole=0.5, l=1.5)
        controller = MPPIController(config_heavy, mppi_config, implementation='numpy')
        
        for _ in range(10):
            result = controller.step()
            assert isinstance(result['force'], float)
            
    def test_state_bounds_enforcement(self, pendulum_config, mppi_config):
        """Тест соблюдения границ состояния"""
        from controller.mppi_controller import MPPIController
        
        # Устанавливаем ограничения
        pendulum_config.max_x = 1.0
        pendulum_config.max_force = 5.0
        
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Искусственно устанавливаем состояние за пределами границ
        controller.state[0] = 2.0  # x за пределами max_x
        
        # Выполняем шаг
        result = controller.step()
        
        # Проверяем, что состояние ограничено
        assert abs(result['state'][0]) <= pendulum_config.max_x + 0.1  # С небольшим запасом
        assert abs(result['force']) <= pendulum_config.max_force + 0.1