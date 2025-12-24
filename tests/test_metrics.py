"""
Тесты для MetricsCalculator
"""
import numpy as np
import pytest


class TestMetricsCalculator:
    """Тесты MetricsCalculator"""
    
    def setup_method(self):
        from src.utils.metrics import MetricsCalculator
        self.calculator = MetricsCalculator
    
    def test_calculate_stability_metrics_basic(self):
        """Базовый тест вычисления метрик устойчивости"""
        states = np.array([
            [0.0, 0.1, 0.0, 0.0],
            [0.1, 0.2, 0.1, 0.1],
            [0.0, 0.15, 0.0, 0.05],
        ])
        
        metrics = self.calculator.calculate_stability_metrics(states)
        
        assert isinstance(metrics, dict)
        assert 'angle_rmse' in metrics
        assert 'angle_std' in metrics
        assert 'max_angle' in metrics
        assert 'position_rmse' in metrics
        assert 'settling_time' in metrics
        
        assert metrics['max_angle'] == 0.2
        assert metrics['angle_rmse'] > 0
    
    def test_calculate_stability_metrics_stable_system(self):
        """Тест для стабильной системы"""
        # Система, которая быстро стабилизировалась
        states = np.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0],
            [0.0, 0.05, 0.0, 0.0],
            [0.0, 0.01, 0.0, 0.0],
            [0.0, 0.005, 0.0, 0.0],
        ])
        
        metrics = self.calculator.calculate_stability_metrics(states)
        
        assert metrics['settling_time'] >= 0
        assert metrics['max_angle'] == 0.5
    
    def test_calculate_control_metrics_basic(self):
        """Базовый тест вычисления метрик управления"""
        controls = np.array([1.0, -2.0, 3.0, -1.0, 0.5])
        
        metrics = self.calculator.calculate_control_metrics(controls)
        
        assert isinstance(metrics, dict)
        assert 'control_effort' in metrics
        assert 'avg_control' in metrics
        assert 'max_control' in metrics
        assert 'control_variance' in metrics
        
        assert metrics['max_control'] == 3.0
        assert metrics['control_effort'] == 7.5
        assert metrics['avg_control'] == 1.5
    
    def test_calculate_control_metrics_zero_controls(self):
        """Тест с нулевыми управлениями"""
        controls = np.zeros(10)
        
        metrics = self.calculator.calculate_control_metrics(controls)
        
        assert metrics['control_effort'] == 0.0
        assert metrics['avg_control'] == 0.0
        assert metrics['max_control'] == 0.0
        assert metrics['control_variance'] == 0.0
    
    def test_calculate_comprehensive_metrics(self):
        """Тест вычисления комплексных метрик"""
        history = {
            'states': [
                np.array([0.0, 0.1, 0.0, 0.0]),
                np.array([0.1, 0.05, 0.1, 0.0]),
                np.array([0.05, 0.02, 0.05, 0.0]),
            ],
            'controls': [1.0, -0.5, 0.2],
            'costs': [0.5, 0.3, 0.1]
        }
        
        metrics = self.calculator.calculate_comprehensive_metrics(history)
        
        assert isinstance(metrics, dict)
        
        # Проверяем наличие всех типов метрик
        stability_keys = ['angle_rmse', 'angle_std', 'max_angle', 'position_rmse', 'settling_time']
        control_keys = ['control_effort', 'avg_control', 'max_control', 'control_variance']
        combined_keys = ['performance_index', 'stability_margin', 'efficiency']
        
        for key in stability_keys + control_keys + combined_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
    
    def test_find_settling_time(self):
        """Тест поиска времени установления"""
        signal = np.array([0.5, 0.3, 0.15, 0.08, 0.05, 0.03, 0.02])
        
        settling_time = self.calculator._find_settling_time(signal, threshold=0.1)
        
        # Порог 0.1, поэтому после индекса 3 все значения < 0.1
        assert settling_time == 3.0
    
    def test_find_settling_time_no_settling(self):
        """Тест когда система не стабилизируется"""
        signal = np.array([0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])
        
        settling_time = self.calculator._find_settling_time(signal, threshold=0.1)
        
        # Ни одно значение не ниже порога 0.1
        assert settling_time == float(len(signal))
    
    def test_find_settling_time_empty_signal(self):
        """Тест с пустым сигналом"""
        signal = np.array([])
        
        settling_time = self.calculator._find_settling_time(signal)
        
        assert settling_time == 0.0
    
    def test_positive_metrics_values(self):
        """Тест что все метрики положительные"""
        states = np.array([
            [0.1, 0.2, 0.05, 0.1],
            [0.0, 0.1, 0.0, 0.05],
            [-0.1, -0.05, -0.1, 0.0],
        ])
        
        controls = np.array([1.0, -0.5, 0.3])
        
        stability_metrics = self.calculator.calculate_stability_metrics(states)
        control_metrics = self.calculator.calculate_control_metrics(controls)
        
        # Проверяем положительность метрик
        for key, value in stability_metrics.items():
            if key != 'settling_time':  # settling_time может быть 0
                assert value >= 0
        
        for key, value in control_metrics.items():
            if key != 'control_variance':  # Дисперсия может быть 0
                assert value >= 0