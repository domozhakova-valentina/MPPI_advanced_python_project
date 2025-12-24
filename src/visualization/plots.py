import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import pandas as pd
from pathlib import Path
from enum import Enum
import warnings
from scipy import stats

from ..controller.config import State, SystemConfig, MPPIConfig
from ..utils.results_collector import ResultsCollector, ResultEntry
from ..utils.metrics import MetricCalculator, PerformanceMetrics


class PlotStyle(Enum):
    """Стили графиков"""
    DEFAULT = "default"  # Стандартный стиль matplotlib
    SEABORN = "seaborn"  # Стиль seaborn
    SCIENTIFIC = "scientific"  # Научный стиль
    MINIMAL = "minimal"  # Минималистичный стиль
    DARK = "dark"  # Темная тема


@dataclass
class PlotConfig:
    """Конфигурация графиков"""
    style: PlotStyle = PlotStyle.SEABORN
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 100
    save_path: Optional[str] = None
    show_grid: bool = True
    tight_layout: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    color_palette: str = "husl"
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8

    def __post_init__(self):
        """Применяет стиль"""
        self.apply_style()

    def apply_style(self):
        """Применяет выбранный стиль"""
        if self.style == PlotStyle.SEABORN:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette(self.color_palette)
        elif self.style == PlotStyle.SCIENTIFIC:
            plt.style.use('seaborn-v0_8-paper')
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': 'Times New Roman',
                'mathtext.fontset': 'stix',
            })
        elif self.style == PlotStyle.MINIMAL:
            plt.style.use('seaborn-v0_8-white')
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
            })
        elif self.style == PlotStyle.DARK:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')


class PlotBuilder:
    """Строитель графиков

    Паттерн: Builder - поэтапное построение сложных графиков
    """

    def __init__(self, config: Optional[PlotConfig] = None):
        """Инициализирует строитель графиков

        Args:
            config: конфигурация графиков
        """
        self.config = config or PlotConfig()
        self.fig = None
        self.axes = None

    def create_figure(self, nrows: int = 1, ncols: int = 1,
                      **kwargs) -> 'PlotBuilder':
        """Создает фигуру

        Args:
            nrows: количество строк
            ncols: количество столбцов
            **kwargs: дополнительные аргументы для subplots

        Returns:
            self для цепочки вызовов
        """
        figsize = kwargs.pop('figsize', self.config.figsize)
        self.fig, self.axes = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=figsize,
            dpi=self.config.dpi,
            **kwargs
        )

        # Делаем axes всегда списком для удобства
        if nrows == 1 and ncols == 1:
            self.axes = np.array([self.axes])
        elif nrows == 1 or ncols == 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = self.axes.flatten()

        return self

    def set_title(self, title: str, ax_index: int = 0, **kwargs):
        """Устанавливает заголовок

        Args:
            title: заголовок
            ax_index: индекс оси
            **kwargs: дополнительные аргументы для set_title
        """
        fontsize = kwargs.pop('fontsize', self.config.title_fontsize)
        self.axes[ax_index].set_title(title, fontsize=fontsize, **kwargs)
        return self

    def set_labels(self, xlabel: str, ylabel: str,
                   ax_index: int = 0, **kwargs):
        """Устанавливает подписи осей

        Args:
            xlabel: подпись оси X
            ylabel: подпись оси Y
            ax_index: индекс оси
            **kwargs: дополнительные аргументы
        """
        fontsize = kwargs.pop('fontsize', self.config.label_fontsize)
        self.axes[ax_index].set_xlabel(xlabel, fontsize=fontsize, **kwargs)
        self.axes[ax_index].set_ylabel(ylabel, fontsize=fontsize, **kwargs)
        return self

    def set_legend(self, ax_index: int = 0, **kwargs):
        """Добавляет легенду

        Args:
            ax_index: индекс оси
            **kwargs: дополнительные аргументы для legend
        """
        fontsize = kwargs.pop('fontsize', self.config.legend_fontsize)
        loc = kwargs.pop('loc', 'best')
        self.axes[ax_index].legend(fontsize=fontsize, loc=loc, **kwargs)
        return self

    def plot_trajectory(self, states: List[State],
                        time_steps: Optional[List[float]] = None,
                        variables: List[str] = None,
                        ax_index: int = 0,
                        **kwargs) -> 'PlotBuilder':
        """Строит график траектории

        Args:
            states: список состояний
            time_steps: временные шаги
            variables: переменные для отображения
            ax_index: индекс оси
            **kwargs: дополнительные аргументы для plot

        Returns:
            self для цепочки вызовов
        """
        if not states:
            return self

        if variables is None:
            variables = ['x', 'theta', 'x_dot', 'theta_dot']

        if time_steps is None:
            time_steps = list(range(len(states)))

        ax = self.axes[ax_index]

        # Преобразуем состояния в словарь массивов
        data = {var: [] for var in variables}
        for state in states:
            if 'x' in variables:
                data['x'].append(state.x)
            if 'theta' in variables:
                data['theta'].append(state.theta)
            if 'x_dot' in variables:
                data['x_dot'].append(state.x_dot)
            if 'theta_dot' in variables:
                data['theta_dot'].append(state.theta_dot)

        # Строим графики
        linewidth = kwargs.pop('linewidth', self.config.line_width)

        labels = {
            'x': 'Положение тележки (м)',
            'theta': 'Угол маятника (рад)',
            'x_dot': 'Скорость тележки (м/с)',
            'theta_dot': 'Угловая скорость (рад/с)'
        }

        for var in variables:
            if var in data and data[var]:
                ax.plot(time_steps, data[var],
                        label=labels.get(var, var),
                        linewidth=linewidth,
                        alpha=self.config.alpha,
                        **kwargs)

        ax.grid(self.config.show_grid, alpha=0.3)
        self.set_labels('Время (с)', 'Значение', ax_index)
        self.set_legend(ax_index)

        return self

    def plot_controls(self, controls: List[float],
                      time_steps: Optional[List[float]] = None,
                      ax_index: int = 0,
                      **kwargs) -> 'PlotBuilder':
        """Строит график управления

        Args:
            controls: список управлений
            time_steps: временные шаги
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        if not controls:
            return self

        if time_steps is None:
            time_steps = list(range(len(controls)))

        ax = self.axes[ax_index]

        linewidth = kwargs.pop('linewidth', self.config.line_width)

        ax.plot(time_steps, controls,
                linewidth=linewidth,
                color='red',
                alpha=self.config.alpha,
                label='Управление',
                **kwargs)

        # Заполняем область между графиком и осью X
        ax.fill_between(time_steps, 0, controls,
                        alpha=0.3, color='red')

        ax.grid(self.config.show_grid, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')

        self.set_labels('Время (с)', 'Сила (Н)', ax_index)
        ax.legend(fontsize=self.config.legend_fontsize)

        return self

    def plot_costs(self, costs: List[float],
                   time_steps: Optional[List[float]] = None,
                   cumulative: bool = False,
                   ax_index: int = 0,
                   **kwargs) -> 'PlotBuilder':
        """Строит график стоимости

        Args:
            costs: список стоимостей
            time_steps: временные шаги
            cumulative: показывать кумулятивную стоимость
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        if not costs:
            return self

        if time_steps is None:
            time_steps = list(range(len(costs)))

        ax = self.axes[ax_index]

        linewidth = kwargs.pop('linewidth', self.config.line_width)

        if cumulative:
            cum_costs = np.cumsum(costs)
            ax.plot(time_steps, cum_costs,
                    linewidth=linewidth,
                    color='green',
                    alpha=self.config.alpha,
                    label='Накопленная стоимость',
                    **kwargs)
            ylabel = 'Накопленная стоимость'
        else:
            ax.plot(time_steps, costs,
                    linewidth=linewidth,
                    color='green',
                    alpha=self.config.alpha,
                    label='Стоимость',
                    **kwargs)
            ylabel = 'Стоимость'

        ax.grid(self.config.show_grid, alpha=0.3)
        self.set_labels('Время (с)', ylabel, ax_index)
        ax.legend(fontsize=self.config.legend_fontsize)

        return self

    def plot_phase_portrait(self, states: List[State],
                            x_var: str = 'theta',
                            y_var: str = 'theta_dot',
                            ax_index: int = 0,
                            **kwargs) -> 'PlotBuilder':
        """Строит фазовый портрет

        Args:
            states: список состояний
            x_var: переменная для оси X
            y_var: переменная для оси Y
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        if not states:
            return self

        ax = self.axes[ax_index]

        # Извлекаем данные
        x_data = []
        y_data = []

        for state in states:
            if x_var == 'x':
                x_data.append(state.x)
            elif x_var == 'theta':
                x_data.append(state.theta)
            elif x_var == 'x_dot':
                x_data.append(state.x_dot)
            elif x_var == 'theta_dot':
                x_data.append(state.theta_dot)

            if y_var == 'x':
                y_data.append(state.x)
            elif y_var == 'theta':
                y_data.append(state.theta)
            elif y_var == 'x_dot':
                y_data.append(state.x_dot)
            elif y_var == 'theta_dot':
                y_data.append(state.theta_dot)

        # Строим фазовый портрет
        linewidth = kwargs.pop('linewidth', self.config.line_width)

        # Основная линия
        ax.plot(x_data, y_data,
                linewidth=linewidth,
                alpha=self.config.alpha,
                **kwargs)

        # Стрелки направления
        if len(x_data) > 10:
            # Выбираем точки для стрелок
            arrow_indices = np.linspace(0, len(x_data) - 2, 5, dtype=int)
            for i in arrow_indices:
                dx = x_data[i + 1] - x_data[i]
                dy = y_data[i + 1] - y_data[i]
                ax.arrow(x_data[i], y_data[i], dx, dy,
                         head_width=0.05, head_length=0.1,
                         fc='red', ec='red', alpha=0.7)

        # Начальная и конечная точки
        ax.scatter(x_data[0], y_data[0],
                   color='green', s=100, label='Начало', zorder=5)
        ax.scatter(x_data[-1], y_data[-1],
                   color='red', s=100, label='Конец', zorder=5)

        labels = {
            'x': 'Положение тележки (м)',
            'theta': 'Угол маятника (рад)',
            'x_dot': 'Скорость тележки (м/с)',
            'theta_dot': 'Угловая скорость (рад/с)'
        }

        ax.grid(self.config.show_grid, alpha=0.3)
        self.set_labels(labels.get(x_var, x_var),
                        labels.get(y_var, y_var), ax_index)
        ax.legend(fontsize=self.config.legend_fontsize)

        return self

    def plot_metrics(self, metrics: Dict[str, float],
                     categories: Optional[Dict[str, List[str]]] = None,
                     ax_index: int = 0,
                     **kwargs) -> 'PlotBuilder':
        """Строит график метрик

        Args:
            metrics: словарь метрик
            categories: категории метрик
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        if not metrics:
            return self

        ax = self.axes[ax_index]

        # Группируем метрики по категориям
        if categories is None:
            categories = {
                'Время': ['avg_compute_time', 'fps', 'total_compute_time'],
                'Управление': ['control_effort', 'max_control'],
                'Состояние': ['max_angle', 'settling_time', 'overshoot'],
                'Стоимость': ['avg_cost', 'total_cost'],
                'Успешность': ['success']
            }

        # Подготавливаем данные
        category_data = {}
        for category, metric_names in categories.items():
            category_values = []
            category_labels = []

            for name in metric_names:
                if name in metrics:
                    category_values.append(metrics[name])

                    # Форматируем метки
                    labels_map = {
                        'avg_compute_time': 'Ср. время (мс)',
                        'fps': 'FPS',
                        'total_compute_time': 'Общ. время (с)',
                        'control_effort': 'Энергия упр.',
                        'max_control': 'Макс. управление (Н)',
                        'max_angle': 'Макс. угол (°)',
                        'settling_time': 'Время уст. (с)',
                        'overshoot': 'Перерег. (%)',
                        'avg_cost': 'Ср. стоимость',
                        'total_cost': 'Общ. стоимость',
                        'success': 'Успешность'
                    }

                    category_labels.append(labels_map.get(name, name))

            if category_values:
                category_data[category] = (category_values, category_labels)

        # Строим grouped bar chart
        if category_data:
            x = np.arange(len(category_data))
            width = 0.8 / max(len(v[0]) for v in category_data.values())

            colors = plt.cm.Set3(np.linspace(0, 1,
                                             max(len(v[0]) for v in category_data.values())))

            for i, (category, (values, labels)) in enumerate(category_data.items()):
                for j, (value, label, color) in enumerate(zip(values, labels, colors)):
                    offset = (j - len(values) / 2 + 0.5) * width
                    ax.bar(x[i] + offset, value, width,
                           label=label if i == 0 else '',
                           color=color, alpha=self.config.alpha)

            ax.set_xticks(x)
            ax.set_xticklabels(list(category_data.keys()))
            ax.grid(self.config.show_grid, alpha=0.3, axis='y')
            ax.legend(fontsize=self.config.legend_fontsize)
            self.set_labels('Категории', 'Значение', ax_index)

        return self

    def plot_comparison(self, results: Dict[str, Dict[str, Any]],
                        metrics: List[str] = None,
                        plot_type: str = 'bar',
                        ax_index: int = 0,
                        **kwargs) -> 'PlotBuilder':
        """Строит график сравнения разных реализаций

        Args:
            results: результаты сравнения
            metrics: метрики для сравнения
            plot_type: тип графика (bar, box, violin)
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        if not results:
            return self

        ax = self.axes[ax_index]

        # Выбираем метрики для сравнения
        if metrics is None:
            metrics = ['success', 'avg_compute_time', 'max_angle', 'fps']

        # Подготавливаем данные
        implementations = list(results.keys())
        metric_data = {metric: [] for metric in metrics}

        for impl in implementations:
            impl_data = results[impl]
            for metric in metrics:
                if metric in impl_data:
                    metric_data[metric].append(impl_data[metric])
                else:
                    metric_data[metric].append(0.0)

        x = np.arange(len(implementations))
        width = 0.8 / len(metrics)

        # Строим график
        if plot_type == 'bar':
            colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))

            for i, (metric, values) in enumerate(metric_data.items()):
                offset = (i - len(metrics) / 2 + 0.5) * width
                ax.bar(x + offset, values, width,
                       label=metric, color=colors[i],
                       alpha=self.config.alpha)

            ax.set_xticks(x)
            ax.set_xticklabels(implementations)
            ax.legend(fontsize=self.config.legend_fontsize)

        elif plot_type == 'box':
            # Box plot для каждой реализации
            box_data = []
            labels = []

            for impl in implementations:
                impl_metrics = []
                for metric in metrics:
                    if metric in results[impl]:
                        impl_metrics.append(results[impl][metric])
                if impl_metrics:
                    box_data.append(impl_metrics)
                    labels.append(impl)

            if box_data:
                bp = ax.boxplot(box_data, labels=labels,
                                patch_artist=True,
                                boxprops=dict(facecolor='lightblue', alpha=0.7))

        ax.grid(self.config.show_grid, alpha=0.3)
        self.set_labels('Реализация', 'Значение', ax_index)

        return self

    def plot_heatmap(self, data: np.ndarray,
                     xlabels: Optional[List[str]] = None,
                     ylabels: Optional[List[str]] = None,
                     ax_index: int = 0,
                     **kwargs) -> 'PlotBuilder':
        """Строит тепловую карту

        Args:
            data: матрица данных
            xlabels: метки оси X
            ylabels: метки оси Y
            ax_index: индекс оси
            **kwargs: дополнительные аргументы

        Returns:
            self для цепочки вызовов
        """
        ax = self.axes[ax_index]

        # Строим heatmap
        im = ax.imshow(data, cmap='viridis', aspect='auto', **kwargs)

        # Добавляем цветовую шкалу
        plt.colorbar(im, ax=ax)

        # Добавляем метки
        if xlabels is not None:
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation=45, ha='right')

        if ylabels is not None:
            ax.set_yticks(np.arange(len(ylabels)))
            ax.set_yticklabels(ylabels)

        # Добавляем значения в ячейки
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if data[i, j] > data.max() / 2 else 'black')

        return self

    def save(self, filename: str, **kwargs):
        """Сохраняет график в файл

        Args:
            filename: имя файла
            **kwargs: дополнительные аргументы для savefig
        """
        if self.fig is None:
            raise ValueError("Нет фигуры для сохранения")

        if self.config.tight_layout:
            self.fig.tight_layout()

        dpi = kwargs.pop('dpi', self.config.dpi)
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
        print(f"График сохранен в {filename}")

    def show(self, **kwargs):
        """Показывает график

        Args:
            **kwargs: дополнительные аргументы для show
        """
        if self.fig is None:
            raise ValueError("Нет фигуры для отображения")

        if self.config.tight_layout:
            self.fig.tight_layout()

        plt.show(**kwargs)

    def close(self):
        """Закрывает фигуру"""
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.axes = None


# Функции высокого уровня

def plot_trajectory(states: List[State],
                    time_steps: Optional[List[float]] = None,
                    config: Optional[PlotConfig] = None,
                    **kwargs) -> PlotBuilder:
    """Строит график траектории

    Args:
        states: список состояний
        time_steps: временные шаги
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_trajectory(states, time_steps, **kwargs)
    builder.set_title('Траектория системы')

    return builder


def plot_controls(controls: List[float],
                  time_steps: Optional[List[float]] = None,
                  config: Optional[PlotConfig] = None,
                  **kwargs) -> PlotBuilder:
    """Строит график управления

    Args:
        controls: список управлений
        time_steps: временные шаги
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_controls(controls, time_steps, **kwargs)
    builder.set_title('Управляющее воздействие')

    return builder


def plot_costs(costs: List[float],
               time_steps: Optional[List[float]] = None,
               cumulative: bool = False,
               config: Optional[PlotConfig] = None,
               **kwargs) -> PlotBuilder:
    """Строит график стоимости

    Args:
        costs: список стоимостей
        time_steps: временные шаги
        cumulative: показывать кумулятивную стоимость
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_costs(costs, time_steps, cumulative, **kwargs)

    title = 'Накопленная стоимость' if cumulative else 'Функция стоимости'
    builder.set_title(title)

    return builder


def plot_comparison(results_collector: ResultsCollector,
                    experiment_name: str,
                    metrics: List[str] = None,
                    config: Optional[PlotConfig] = None,
                    **kwargs) -> PlotBuilder:
    """Строит график сравнения реализаций

    Args:
        results_collector: коллектор результатов
        experiment_name: имя эксперимента
        metrics: метрики для сравнения
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    # Получаем результаты эксперимента
    experiment = results_collector.get_experiment(experiment_name)
    if not experiment:
        raise ValueError(f"Эксперимент '{experiment_name}' не найден")

    # Группируем результаты по реализации
    results_by_impl = {}
    for result in experiment.results:
        impl = result.implementation
        if impl not in results_by_impl:
            results_by_impl[impl] = []
        results_by_impl[impl].append(result)

    # Вычисляем средние метрики для каждой реализации
    comparison_data = {}
    for impl, results in results_by_impl.items():
        if results:
            # Используем метрики из первого результата если есть
            if results[0].metrics:
                comparison_data[impl] = results[0].metrics
            else:
                # Или вычисляем базовые метрики
                metrics_calc = MetricCalculator()
                all_metrics = []
                for result in results:
                    data = {
                        'states': result.states,
                        'controls': result.controls,
                        'costs': result.costs,
                        'compute_times': result.compute_times,
                        'time_steps': result.time_steps,
                        'success': result.success
                    }
                    metrics = metrics_calc.compute_all(data)
                    all_metrics.append(metrics)

                # Усредняем метрики
                if all_metrics:
                    avg_metrics = {}
                    for metric_name in all_metrics[0].keys():
                        values = [m[metric_name] for m in all_metrics
                                  if metric_name in m]
                        if values:
                            avg_metrics[metric_name] = np.mean(values)
                    comparison_data[impl] = avg_metrics

    # Строим график
    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_comparison(comparison_data, metrics, **kwargs)
    builder.set_title(f'Сравнение реализаций: {experiment_name}')

    return builder


def plot_metrics(metrics: Dict[str, float],
                 config: Optional[PlotConfig] = None,
                 **kwargs) -> PlotBuilder:
    """Строит график метрик

    Args:
        metrics: словарь метрик
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_metrics(metrics, **kwargs)
    builder.set_title('Метрики производительности')

    return builder


def plot_phase_portrait(states: List[State],
                        x_var: str = 'theta',
                        y_var: str = 'theta_dot',
                        config: Optional[PlotConfig] = None,
                        **kwargs) -> PlotBuilder:
    """Строит фазовый портрет

    Args:
        states: список состояний
        x_var: переменная для оси X
        y_var: переменная для оси Y
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_phase_portrait(states, x_var, y_var, **kwargs)

    labels = {
        'theta': 'Угол (рад)',
        'theta_dot': 'Угловая скорость (рад/с)',
        'x': 'Положение (м)',
        'x_dot': 'Скорость (м/с)'
    }

    title = f'Фазовый портрет: {labels.get(x_var, x_var)} vs {labels.get(y_var, y_var)}'
    builder.set_title(title)

    return builder


def plot_heatmap(data: np.ndarray,
                 xlabels: Optional[List[str]] = None,
                 ylabels: Optional[List[str]] = None,
                 title: str = "Тепловая карта",
                 config: Optional[PlotConfig] = None,
                 **kwargs) -> PlotBuilder:
    """Строит тепловую карту

    Args:
        data: матрица данных
        xlabels: метки оси X
        ylabels: метки оси Y
        title: заголовок
        config: конфигурация графиков
        **kwargs: дополнительные аргументы

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(**{k: v for k, v in kwargs.items()
                               if k in PlotConfig.__annotations__})

    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_heatmap(data, xlabels, ylabels, **kwargs)
    builder.set_title(title)

    return builder


def create_dashboard(trajectory: List[State],
                     controls: List[float],
                     costs: List[float],
                     time_steps: Optional[List[float]] = None,
                     config: Optional[PlotConfig] = None) -> PlotBuilder:
    """Создает дашборд с несколькими графиками

    Args:
        trajectory: траектория состояний
        controls: управления
        costs: стоимости
        time_steps: временные шаги
        config: конфигурация графиков

    Returns:
        строитель графиков
    """
    if config is None:
        config = PlotConfig(style=PlotStyle.SEABORN, figsize=(15, 10))

    builder = PlotBuilder(config)
    builder.create_figure(2, 2)

    # График 1: Траектория
    builder.plot_trajectory(trajectory, time_steps, ax_index=0)
    builder.set_title('Траектория системы', ax_index=0)

    # График 2: Управление
    builder.plot_controls(controls, time_steps, ax_index=1)
    builder.set_title('Управляющее воздействие', ax_index=1)

    # График 3: Стоимость
    builder.plot_costs(costs, time_steps, ax_index=2)
    builder.set_title('Функция стоимости', ax_index=2)

    # График 4: Фазовый портрет
    builder.plot_phase_portrait(trajectory, ax_index=3)
    builder.set_title('Фазовый портрет', ax_index=3)

    builder.fig.suptitle('Дашборд результатов MPPI', fontsize=16)

    return builder


def create_comparison_report(collector: ResultsCollector,
                             experiment_names: List[str],
                             output_dir: str = "reports"):
    """Создает отчет сравнения экспериментов

    Args:
        collector: коллектор результатов
        experiment_names: имена экспериментов
        output_dir: директория для сохранения
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    config = PlotConfig(style=PlotStyle.SCIENTIFIC, figsize=(12, 8))

    # Собираем данные для сравнения
    comparison_data = {}

    for exp_name in experiment_names:
        experiment = collector.get_experiment(exp_name)
        if experiment and experiment.results:
            # Берем первый результат для примера
            result = experiment.results[0]
            if result.metrics:
                comparison_data[exp_name] = result.metrics

    if not comparison_data:
        print("Нет данных для сравнения")
        return

    # 1. График сравнения метрик
    builder = PlotBuilder(config)
    builder.create_figure()
    builder.plot_comparison(comparison_data, plot_type='bar')
    builder.set_title('Сравнение экспериментов')
    builder.save(output_dir / "comparison_bar.png")
    builder.close()

    # 2. Heatmap метрик
    # Преобразуем в матрицу
    all_metrics = set()
    for metrics in comparison_data.values():
        all_metrics.update(metrics.keys())

    all_metrics = sorted(list(all_metrics))
    experiments = list(comparison_data.keys())

    data_matrix = np.zeros((len(experiments), len(all_metrics)))

    for i, exp in enumerate(experiments):
        for j, metric in enumerate(all_metrics):
            data_matrix[i, j] = comparison_data[exp].get(metric, 0)

    # Нормализуем по столбцам для лучшей визуализации
    data_normalized = (data_matrix - data_matrix.min(axis=0)) / \
                      (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)

    builder = PlotBuilder(config)
    builder.create_figure(figsize=(12, 6))
    builder.plot_heatmap(data_normalized, all_metrics, experiments,
                         title='Нормализованные метрики экспериментов')
    builder.save(output_dir / "comparison_heatmap.png")
    builder.close()

    # 3. Создаем HTML отчет
    html_report = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет сравнения экспериментов</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .image-container {{
                display: flex;
                flex-direction: column;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .image-container img {{
                width: 100%;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 30px;
            }}
            th, td {{
                padding: 12px;
                text-align: center;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            .timestamp {{
                color: #888;
                text-align: right;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Отчет сравнения экспериментов MPPI</h1>
            <p>Сравнение {len(experiments)} экспериментов по {len(all_metrics)} метрикам</p>
        </div>

        <div class="image-container">
            <h2>Сравнение экспериментов</h2>
            <img src="comparison_bar.png" alt="Сравнение экспериментов">

            <h2>Тепловая карта метрик</h2>
            <img src="comparison_heatmap.png" alt="Тепловая карта метрик">
        </div>

        <h2>Детальные данные</h2>
        <table>
            <thead>
                <tr>
                    <th>Эксперимент</th>
    """

    for metric in all_metrics:
        html_report += f'<th>{metric}</th>'

    html_report += """
                </tr>
            </thead>
            <tbody>
    """

    for exp in experiments:
        html_report += f'<tr><td>{exp}</td>'
        for metric in all_metrics:
            value = comparison_data[exp].get(metric, 0)
            html_report += f'<td>{value:.4f}</td>'
        html_report += '</tr>'

    html_report += f"""
            </tbody>
        </table>

        <div class="timestamp">
            Отчет создан: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </body>
    </html>
    """

    with open(output_dir / "comparison_report.html", 'w', encoding='utf-8') as f:
        f.write(html_report)

    print(f"Отчет сохранен в {output_dir}")


# Пример использования
if __name__ == "__main__":
    print("Testing Plots Module")
    print("=" * 60)

    # Создаем тестовые данные
    num_points = 100
    time = np.linspace(0, 10, num_points)

    trajectory = []
    controls = []
    costs = []

    for t in time:
        state = State(
            x=0.5 * np.sin(t),
            theta=0.3 * np.sin(2 * t),
            x_dot=0.5 * np.cos(t),
            theta_dot=0.6 * np.cos(2 * t)
        )
        trajectory.append(state)
        controls.append(2.0 * np.sin(t))
        costs.append(10.0 * np.sin(t / 2) ** 2)

    print(f"Создано {len(trajectory)} точек данных")

    # Тестируем разные типы графиков
    test_functions = [
        ("plot_trajectory", lambda: plot_trajectory(trajectory, time)),
        ("plot_controls", lambda: plot_controls(controls, time)),
        ("plot_costs", lambda: plot_costs(costs, time)),
        ("plot_phase_portrait", lambda: plot_phase_portrait(trajectory)),
    ]

    for func_name, func in test_functions:
        print(f"\nТестируем {func_name}...")

        try:
            builder = func()
            builder.save(f"test_{func_name}.png", dpi=100)
            builder.close()
            print(f"  ✓ График сохранен как test_{func_name}.png")

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

    # Тестируем дашборд
    print("\nТестируем создание дашборда...")
    try:
        dashboard = create_dashboard(trajectory, controls, costs, time)
        dashboard.save("test_dashboard.png", dpi=150)
        dashboard.close()
        print("  ✓ Дашборд сохранен как test_dashboard.png")
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")

    # Тестируем разные стили
    print("\nТестируем разные стили графиков...")
    for style in PlotStyle:
        print(f"  Стиль {style.value}: ", end="")

        try:
            config = PlotConfig(style=style, figsize=(8, 4))
            builder = PlotBuilder(config)
            builder.create_figure()
            builder.plot_trajectory(trajectory, time)
            builder.set_title(f'Стиль: {style.value}')
            builder.save(f"test_style_{style.value}.png")
            builder.close()
            print("✓")

        except Exception as e:
            print(f"✗ ({e})")

    print("\n" + "=" * 60)
    print("Plots module tested successfully!")
    print("=" * 60)