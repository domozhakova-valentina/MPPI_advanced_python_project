"""
Вычисление метрик производительности
"""
import numpy as np
from typing import Dict, List, Any


class MetricsCalculator:
    """Калькулятор метрик производительности"""
    
    @staticmethod
    def calculate_stability_metrics(states: np.ndarray) -> Dict[str, float]:
        """
        Вычисление метрик устойчивости
        
        Args:
            states: массив состояний размером (N, 4)
            
        Returns:
            Словарь с метриками устойчивости
        """
        angles = states[:, 1]  # углы маятника
        positions = states[:, 0]  # положения тележки
        
        return {
            'angle_rmse': np.sqrt(np.mean(angles**2)),
            'angle_std': np.std(angles),
            'max_angle': np.max(np.abs(angles)),
            'position_rmse': np.sqrt(np.mean(positions**2)),
            'settling_time': MetricsCalculator._find_settling_time(angles)
        }
    
    @staticmethod
    def _find_settling_time(signal: np.ndarray, 
                          threshold: float = 0.1) -> float:
        """
        Поиск времени установления
        
        Args:
            signal: сигнал
            threshold: порог установления
            
        Returns:
            Время установления в шагах
        """
        abs_signal = np.abs(signal)
        for i in range(len(abs_signal)):
            if np.all(abs_signal[i:] < threshold):
                return float(i)
        return float(len(signal))
    
    @staticmethod
    def calculate_control_metrics(controls: np.ndarray) -> Dict[str, float]:
        """
        Вычисление метрик управления
        
        Args:
            controls: массив управляющих воздействий
            
        Returns:
            Словарь с метриками управления
        """
        return {
            'control_effort': np.sum(np.abs(controls)),
            'avg_control': np.mean(np.abs(controls)),
            'max_control': np.max(np.abs(controls)),
            'control_variance': np.var(controls)
        }
    
    @staticmethod
    def calculate_comprehensive_metrics(history: Dict[str, List]) -> Dict[str, float]:
        """
        Вычисление комплексных метрик
        
        Args:
            history: словарь с историей
            
        Returns:
            Словарь со всеми метриками
        """
        states = np.array(history['states'])
        controls = np.array(history['controls'])
        
        stability = MetricsCalculator.calculate_stability_metrics(states)
        control = MetricsCalculator.calculate_control_metrics(controls)
        
        # Комбинированные метрики
        combined = {
            'performance_index': stability['angle_rmse'] + 0.1 * control['control_effort'],
            'stability_margin': 1.0 / (stability['angle_rmse'] + 1e-6),
            'efficiency': 1.0 / (control['control_effort'] + 1e-6)
        }
        
        return {**stability, **control, **combined}