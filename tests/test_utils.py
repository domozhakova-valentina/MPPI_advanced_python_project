"""
Тесты утилит
"""
import numpy as np
import pytest


class TestResultsCollector:
    """Тесты ResultsCollector"""
    
    def test_initialization(self):
        """Тест инициализации ResultsCollector"""
        from utils.results_collector import ResultsCollector
        
        collector = ResultsCollector()
        
        assert collector.results == {}
        assert collector.comparison_data == {}
        
    def test_run_simulation(self, pendulum_config, mppi_config):
        """Тест запуска симуляции через ResultsCollector"""
        from utils.results_collector import ResultsCollector
        from controller.mppi_controller import MPPIController
        
        collector = ResultsCollector()
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        
        # Запускаем симуляцию
        result = collector.run_simulation(
            controller=controller,
            num_steps=10,
            experiment_name='test_experiment'
        )
        
        # Проверяем результат
        assert isinstance(result, dict)
        assert 'test_experiment' in collector.results
        
        # Проверяем структуру результата
        assert 'history' in result
        assert 'metrics' in result
        assert 'config' in result
        
        # Проверяем метрики
        metrics = result['metrics']
        assert 'execution_time' in metrics
        assert metrics['execution_time'] > 0
        assert metrics['num_steps'] == 10
        
    def test_run_comparison(self, pendulum_config, mppi_config):
        """Тест сравнения нескольких реализаций"""
        from utils.results_collector import ResultsCollector
        from controller.mppi_controller import MPPIController
        
        collector = ResultsCollector()
        
        # Создаем контроллеры (только numpy для теста)
        controllers = {
            'numpy_1': MPPIController(pendulum_config, mppi_config, implementation='numpy'),
            'numpy_2': MPPIController(pendulum_config, mppi_config, implementation='numpy')
        }
        
        # Запускаем сравнение
        results = collector.run_comparison(controllers, num_steps=5)
        
        # Проверяем результаты
        assert isinstance(results, dict)
        assert 'numpy_1' in results
        assert 'numpy_2' in results
        assert 'numpy_1' in collector.comparison_data
        assert 'numpy_2' in collector.comparison_data
        
    def test_get_comparison_summary(self, pendulum_config, mppi_config):
        """Тест получения сводки сравнения"""
        from utils.results_collector import ResultsCollector
        from controller.mppi_controller import MPPIController
        
        collector = ResultsCollector()
        
        # Создаем контроллеры и запускаем симуляции
        controller = MPPIController(pendulum_config, mppi_config, implementation='numpy')
        collector.run_simulation(controller, num_steps=5, experiment_name='test')
        
        # Получаем сводку
        summary = collector.get_comparison_summary()
        
        # Проверяем результат
        assert isinstance(summary, dict)
        assert 'test' in summary
        assert 'execution_time' in summary['test']


class TestMetricsCalculator:
    """Тесты MetricsCalculator"""
    
    def test_calculate_stability_metrics(self):
        """Тест вычисления метрик устойчивости"""
        from utils.metrics import MetricsCalculator
        
        # Создаем тестовые состояния
        states = np.array([
            [0.0, 0.1, 0.0, 0.0],   # небольшое отклонение
            [0.0, 0.2, 0.0, 0.0],   # большее отклонение
            [0.0, 0.1, 0.0, 0.0],   # возврат
            [0.0, 0.0, 0.0, 0.0]    # идеальное положение
        ])
        
        metrics = MetricsCalculator.calculate_stability_metrics(states)
        
        # Проверяем результат
        assert isinstance(metrics, dict)
        assert 'angle_rmse' in metrics
        assert 'angle_std' in metrics
        assert 'max_angle' in metrics
        assert 'position_rmse' in metrics
        assert 'settling_time' in metrics
        
        # Проверяем значения
        assert metrics['max_angle'] == 0.2
        assert metrics['angle_rmse'] > 0
        assert metrics['settling_time'] >= 0
        
    def test_calculate_control_metrics(self):
        """Тест вычисления метрик управления"""
        from utils.metrics import MetricsCalculator
        
        # Создаем тестовые управления
        controls = np.array([0.1, -0.2, 0.3, -0.1, 0.0])
        
        metrics = MetricsCalculator.calculate_control_metrics(controls)
        
        # Проверяем результат
        assert isinstance(metrics, dict)
        assert 'control_effort' in metrics
        assert 'avg_control' in metrics
        assert 'max_control' in metrics
        assert 'control_variance' in metrics
        
        # Проверяем значения
        assert metrics['control_effort'] > 0
        assert metrics['avg_control'] > 0
        assert metrics['max_control'] == 0.3
        
    def test_calculate_comprehensive_metrics(self, mock_history):
        """Тест вычисления комплексных метрик"""
        from utils.metrics import MetricsCalculator
        
        metrics = MetricsCalculator.calculate_comprehensive_metrics(mock_history)
        
        # Проверяем результат
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Проверяем наличие ключевых метрик
        expected_keys = ['performance_index', 'stability_margin', 'efficiency']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)