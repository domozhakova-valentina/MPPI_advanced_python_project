"""
Сбор и агрегация результатов
"""
import time
from typing import Dict, List, Any, Optional
import numpy as np


try:
    # Если запускаем из корня проекта
    from controller.mppi_controller import MPPIController
except ImportError:
    # Если запускаем из поддиректории
    from ..controller import MPPIController

class ResultsCollector:
    """Класс для сбора и агрегации результатов экспериментов"""
    
    def __init__(self):
        self.results = {}
        self.comparison_data = {}
    
    def run_simulation(self, 
                      controller: MPPIController,
                      num_steps: int = 500,
                      experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Запуск симуляции и сбор результатов
        
        Args:
            controller: контроллер MPPI
            num_steps: количество шагов симуляции
            experiment_name: имя эксперимента
            
        Returns:
            Словарь с результатами
        """
        if experiment_name is None:
            experiment_name = controller.implementation_name
        
        print(f"Запуск симуляции: {experiment_name}")
        
        # Сброс контроллера
        controller.reset()
        
        # Измерение времени выполнения
        start_time = time.time()
        
        # Запуск симуляции
        for step in range(num_steps):
            controller.step()
        
        # Вычисление времени выполнения
        exec_time = time.time() - start_time
        
        # Сбор метрик
        metrics = controller.get_performance_metrics()
        metrics['execution_time'] = exec_time
        metrics['num_steps'] = num_steps
        
        # Сохранение результатов
        self.results[experiment_name] = {
            'history': controller.history.copy(),
            'metrics': metrics,
            'config': {
                'pendulum': controller.pendulum_config,
                'mppi': controller.mppi_config
            }
        }
        
        print(f"Симуляция завершена за {exec_time:.4f} секунд")
        print(f"Метрики: {metrics}")
        
        return self.results[experiment_name]
    
    def run_comparison(self,
                      controllers: Dict[str, MPPIController],
                      num_steps: int = 500) -> Dict[str, Dict[str, Any]]:
        """
        Запуск сравнения нескольких реализаций
        
        Args:
            controllers: словарь {имя: контроллер}
            num_steps: количество шагов симуляции
            
        Returns:
            Словарь с результатами всех экспериментов
        """
        self.comparison_data = {}
        
        for name, controller in controllers.items():
            result = self.run_simulation(controller, num_steps, name)
            self.comparison_data[name] = result
        
        return self.comparison_data
    
    def get_comparison_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Получение сводки сравнения
        
        Returns:
            Словарь с метриками для каждой реализации
        """
        summary = {}
        
        for name, data in self.comparison_data.items():
            summary[name] = data['metrics']
        
        return summary
    
    def save_results(self, filename: str):
        """Сохранение результатов в файл"""
        import pickle
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'comparison': self.comparison_data
            }, f)
    
    def load_results(self, filename: str):
        """Загрузка результатов из файла"""
        import pickle
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.results = data['results']
            self.comparison_data = data['comparison']
