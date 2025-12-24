"""
Построение графиков результатов
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import seaborn as sns


class Plotter:
    """Класс для построения графиков результатов"""
    
    @staticmethod
    def plot_simulation_results(history: Dict[str, List]) -> plt.Figure:
        """
        Построение графиков результатов симуляции
        
        history - это словарь с историей состояний, управлений и стоимостейs:
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        states = np.array(history['states'])
        controls = np.array(history['controls'])
        costs = np.array(history['costs'])
        time = np.arange(len(states))
        
        # 1. Угол маятника
        axes[0, 0].plot(time, states[:, 1], 'b-', lw=2)
        axes[0, 0].set_xlabel('Шаг')
        axes[0, 0].set_ylabel('Угол (рад)')
        axes[0, 0].set_title('Угол маятника')
        axes[0, 0].grid(True)
        
        # 2. Сила управления
        axes[0, 1].plot(time, controls, 'g-', lw=2)
        axes[0, 1].set_xlabel('Шаг')
        axes[0, 1].set_ylabel('Сила (Н)')
        axes[0, 1].set_title('Прикладываемая сила')
        axes[0, 1].grid(True)
        
        # 3. Положение тележки
        axes[0, 2].plot(time, states[:, 0], 'm-', lw=2)
        axes[0, 2].set_xlabel('Шаг')
        axes[0, 2].set_ylabel('Положение (м)')
        axes[0, 2].set_title('Положение тележки')
        axes[0, 2].grid(True)
        
        # 4. Фазовая плоскость угла
        axes[1, 0].plot(states[:, 1], states[:, 3], 'r-', lw=1, alpha=0.7)
        axes[1, 0].set_xlabel('Угол (рад)')
        axes[1, 0].set_ylabel('Угловая скорость (рад/с)')
        axes[1, 0].set_title('Фазовая плоскость угла')
        axes[1, 0].grid(True)
        
        # 5. Стоимость
        axes[1, 1].plot(time, costs, 'k-', lw=2)
        axes[1, 1].set_xlabel('Шаг')
        axes[1, 1].set_ylabel('Стоимость')
        axes[1, 1].set_title('Значение функции стоимости')
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        # 6. Гистограмма управлений
        axes[1, 2].hist(controls, bins=30, alpha=0.7, color='orange')
        axes[1, 2].set_xlabel('Сила (Н)')
        axes[1, 2].set_ylabel('Частота')
        axes[1, 2].set_title('Распределение управляющих воздействий')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> plt.Figure:
        """
        Построение графиков сравнения реализаций
        
         metrics_dict - это словарь {название_реализации: метрики}
        """
        implementations = list(metrics_dict.keys())
        metrics_names = list(metrics_dict[implementations[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics_names), figsize=(15, 4))
        
        if len(metrics_names) == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(implementations)))
        
        for idx, metric in enumerate(metrics_names):
            values = [metrics_dict[impl][metric] for impl in implementations]
            axes[idx].bar(implementations, values, color=colors)
            axes[idx].set_title(metric)
            axes[idx].set_ylabel('Значение')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_performance_comparison(times_dict: Dict[str, float]) -> plt.Figure:
        """
        Построение графика сравнения времени выполнения
        
        times_dict - словарь вида {название_реализации: время_выполнения}
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        implementations = list(times_dict.keys())
        times = list(times_dict.values())
        
        bars = ax.bar(implementations, times, color=['blue', 'green', 'red'])
        
        # Добавление значений на столбцы
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.4f} с', ha='center', va='bottom')
        
        ax.set_xlabel('Реализация')
        ax.set_ylabel('Время выполнения (с)')
        ax.set_title('Сравнение производительности реализаций MPPI')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
