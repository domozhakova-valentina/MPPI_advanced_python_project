"""
Визуализация результатов работы алгоритма MPPI.

Содержит функции для:
1. Анимации движения маятника
2. Построения графиков траекторий
3. Сравнения разных реализаций
4. Создания интерактивных виджетов

Паттерны:
- Builder: построение сложных визуализаций
- Facade: простой интерфейс для сложных графиков
- Observer: обновление графиков в реальном времени
"""

from .animate import (
    PendulumAnimator,
    create_animation,
    save_animation,
    real_time_animation,
    create_interactive_animation
)

from .plots import (
    PlotBuilder,
    plot_trajectory,
    plot_controls,
    plot_costs,
    plot_comparison,
    plot_metrics,
    plot_phase_portrait,
    plot_heatmap,
    create_dashboard,
    create_comparison_report
)

__all__ = [
    # Анимация
    'PendulumAnimator',
    'create_animation',
    'save_animation',
    'real_time_animation',
    'create_interactive_animation',

    # Графики
    'PlotBuilder',
    'plot_trajectory',
    'plot_controls',
    'plot_costs',
    'plot_comparison',
    'plot_metrics',
    'plot_phase_portrait',
    'plot_heatmap',
    'create_dashboard',
    'create_comparison_report'
]

__version__ = '1.0.0'


def print_visualization_info():
    """Выводит информацию о модуле визуализации"""
    print("=" * 60)
    print("MPPI Visualization Module")
    print("=" * 60)
    print("\nДоступные функции:")
    print("  1. Анимация маятника в реальном времени")
    print("  2. Графики траекторий и управлений")
    print("  3. Сравнение производительности реализаций")
    print("  4. Интерактивные виджеты для Jupyter")
    print("=" * 60)


# Автоматически выводим информацию при импорте
print_visualization_info()