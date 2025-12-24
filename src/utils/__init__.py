"""
<<<<<<< HEAD
Утилиты для сбора результатов и вычисления метрик
"""
from .results_collector import ResultsCollector
from .metrics import MetricsCalculator

__all__ = ['ResultsCollector', 'MetricsCalculator']
=======
Утилиты для проекта MPPI.

Содержит:
1. Сборщик результатов для агрегации и анализа данных
2. Метрики для оценки производительности алгоритмов
3. Вспомогательные функции для обработки данных

Паттерны:
- Repository: для хранения и доступа к результатам
- Decorator: для измерения времени и профилирования
- Adapter: для преобразования форматов данных
- Composite: для комбинирования метрик
"""

from .results_collector import (
    ResultsCollector,
    ResultEntry,
    ExperimentRun,
    save_results,
    load_results,
    merge_results,
    filter_results,
    export_to_csv,
    export_to_excel,
    plot_comparison,
    generate_report
)

from .metrics import (
    Metric,
    TimeMetric,
    ControlMetric,
    StateMetric,
    CostMetric,
    CompositeMetric,
    SuccessMetric,
    MetricCalculator,
    calculate_all_metrics,
    compare_metrics,
    create_metric_summary,
    save_metrics_report
)

__all__ = [
    # Results Collector
    'ResultsCollector',
    'ResultEntry',
    'ExperimentRun',
    'save_results',
    'load_results',
    'merge_results',
    'filter_results',
    'export_to_csv',
    'export_to_excel',
    'plot_comparison',
    'generate_report',

    # Metrics
    'Metric',
    'TimeMetric',
    'ControlMetric',
    'StateMetric',
    'CostMetric',
    'CompositeMetric',
    'SuccessMetric',
    'MetricCalculator',
    'calculate_all_metrics',
    'compare_metrics',
    'create_metric_summary',
    'save_metrics_report'
]

__version__ = '1.0.0'


def print_utils_info():
    """Выводит информацию об утилитах"""
    print("=" * 60)
    print("MPPI Utils Package")
    print("=" * 60)
    print("\nДоступные утилиты:")
    print("  1. ResultsCollector - сбор и анализ результатов экспериментов")
    print("  2. MetricCalculator - расчет метрик производительности")
    print("  3. Вспомогательные функции для экспорта и визуализации")
    print("=" * 60)


# Автоматически выводим информацию при импорте
print_utils_info()
>>>>>>> 940c7edbb053fa3bce774f825a702520c53721c0
