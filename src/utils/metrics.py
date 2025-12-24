"""
Метрики для оценки производительности алгоритмов MPPI.

Паттерны:
- Strategy: разные типы метрик с общим интерфейсом
- Composite: комбинированные метрики
- Decorator: метрики с дополнительной функциональностью
- Factory: создание метрик по конфигурации
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from enum import Enum
import math
from datetime import datetime
import json
from scipy import signal  # Для обработки сигналов
import warnings


class MetricType(Enum):
    """Типы метрик"""
    TIME = "time"  # Метрики времени
    CONTROL = "control"  # Метрики управления
    STATE = "state"  # Метрики состояния
    COST = "cost"  # Метрики стоимости
    COMPOSITE = "composite"  # Комбинированные метрики
    SUCCESS = "success"  # Метрики успешности


class AggregationMethod(Enum):
    """Методы агрегации для временных рядов"""
    MEAN = "mean"  # Среднее значение
    SUM = "sum"  # Сумма
    MAX = "max"  # Максимальное значение
    MIN = "min"  # Минимальное значение
    STD = "std"  # Стандартное отклонение
    LAST = "last"  # Последнее значение
    INTEGRAL = "integral"  # Интеграл (сумма с учетом dt)


@dataclass
class MetricConfig:
    """Конфигурация метрики

    Паттерн: Builder - для создания сложных конфигураций метрик
    """
    name: str  # Имя метрики
    metric_type: MetricType  # Тип метрики
    aggregation: AggregationMethod = AggregationMethod.MEAN  # Метод агрегации
    weight: float = 1.0  # Вес метрики в комбинированных оценках
    threshold: Optional[float] = None  # Пороговое значение (для успеха/неудачи)
    higher_is_better: bool = True  # Лучше ли большее значение
    description: Optional[str] = None  # Описание метрики
    units: Optional[str] = None  # Единицы измерения

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            'name': self.name,
            'metric_type': self.metric_type.value,
            'aggregation': self.aggregation.value,
            'weight': self.weight,
            'threshold': self.threshold,
            'higher_is_better': self.higher_is_better,
            'description': self.description,
            'units': self.units
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricConfig':
        """Создает из словаря"""
        return cls(
            name=data['name'],
            metric_type=MetricType(data['metric_type']),
            aggregation=AggregationMethod(data['aggregation']),
            weight=data.get('weight', 1.0),
            threshold=data.get('threshold'),
            higher_is_better=data.get('higher_is_better', True),
            description=data.get('description'),
            units=data.get('units')
        )


class Metric(ABC):
    """Абстрактный базовый класс для метрик

    Паттерн: Strategy - определяет интерфейс для всех метрик
    """

    def __init__(self, config: MetricConfig):
        """Инициализирует метрику

        Args:
            config: конфигурация метрики
        """
        self.config = config
        self.value: Optional[float] = None
        self.computed: bool = False

    @abstractmethod
    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет значение метрики

        Args:
            data: словарь с данными для вычисления

        Returns:
            значение метрики
        """
        pass

    def compute_and_store(self, data: Dict[str, Any]) -> float:
        """Вычисляет и сохраняет значение метрики"""
        self.value = self.compute(data)
        self.computed = True
        return self.value

    def get_value(self) -> Optional[float]:
        """Возвращает вычисленное значение метрики"""
        return self.value

    def get_normalized(self, min_val: float, max_val: float) -> Optional[float]:
        """Возвращает нормализованное значение [0, 1]"""
        if self.value is None:
            return None

        if min_val == max_val:
            return 0.5

        normalized = (self.value - min_val) / (max_val - min_val)

        # Инвертируем если меньше значение лучше
        if not self.config.higher_is_better:
            normalized = 1.0 - normalized

        return max(0.0, min(1.0, normalized))

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует метрику в словарь"""
        return {
            'config': self.config.to_dict(),
            'value': self.value,
            'computed': self.computed
        }

    def __str__(self) -> str:
        """Строковое представление"""
        if self.value is not None:
            units = f" {self.config.units}" if self.config.units else ""
            return f"{self.config.name}: {self.value:.4f}{units}"
        else:
            return f"{self.config.name}: не вычислено"

    def __repr__(self) -> str:
        """Представление для отладки"""
        return f"Metric(name={self.config.name}, value={self.value})"


class TimeMetric(Metric):
    """Метрики времени выполнения"""

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет метрику времени"""
        compute_times = data.get('compute_times', [])

        if not compute_times:
            return 0.0

        compute_array = np.array(compute_times)

        if self.config.name == 'avg_compute_time':
            return float(np.mean(compute_array))

        elif self.config.name == 'total_compute_time':
            return float(np.sum(compute_array))

        elif self.config.name == 'max_compute_time':
            return float(np.max(compute_array))

        elif self.config.name == 'min_compute_time':
            return float(np.min(compute_array))

        elif self.config.name == 'std_compute_time':
            return float(np.std(compute_array))

        elif self.config.name == 'fps':
            avg_time = np.mean(compute_array)
            return 1.0 / avg_time if avg_time > 0 else 0.0

        else:
            raise ValueError(f"Неизвестная метрика времени: {self.config.name}")


class ControlMetric(Metric):
    """Метрики управления"""

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет метрику управления"""
        controls = data.get('controls', [])
        time_steps = data.get('time_steps', [])

        if not controls:
            return 0.0

        control_array = np.array(controls)

        if self.config.name == 'avg_control':
            return float(np.mean(np.abs(control_array)))

        elif self.config.name == 'max_control':
            return float(np.max(np.abs(control_array)))

        elif self.config.name == 'control_variance':
            return float(np.var(control_array))

        elif self.config.name == 'control_effort':
            # Интеграл от квадрата управления
            if len(time_steps) > 1:
                dt = time_steps[1] - time_steps[0] if len(time_steps) > 1 else 1.0
                return float(np.sum(control_array ** 2) * dt)
            else:
                return float(np.sum(control_array ** 2))

        elif self.config.name == 'control_smoothness':
            # Гладкость управления (средняя скорость изменения)
            if len(controls) > 1:
                changes = np.diff(control_array)
                return float(np.mean(np.abs(changes)))
            else:
                return 0.0

        else:
            raise ValueError(f"Неизвестная метрика управления: {self.config.name}")


class StateMetric(Metric):
    """Метрики состояния системы"""

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет метрику состояния"""
        states = data.get('states', [])
        time_steps = data.get('time_steps', [])

        if not states:
            return 0.0

        # Извлекаем нужные компоненты состояния
        if self.config.name.startswith('angle'):
            values = [s.get('theta', 0.0) for s in states]
        elif self.config.name.startswith('position'):
            values = [s.get('x', 0.0) for s in states]
        elif self.config.name.startswith('angular_velocity'):
            values = [s.get('theta_dot', 0.0) for s in states]
        elif self.config.name.startswith('velocity'):
            values = [s.get('x_dot', 0.0) for s in states]
        else:
            values = []

        if not values:
            return 0.0

        values_array = np.array(values)

        if 'avg' in self.config.name:
            return float(np.mean(np.abs(values_array)))

        elif 'max' in self.config.name:
            return float(np.max(np.abs(values_array)))

        elif 'std' in self.config.name:
            return float(np.std(values_array))

        elif 'rms' in self.config.name:  # Root Mean Square
            return float(np.sqrt(np.mean(values_array ** 2)))

        elif 'settling_time' in self.config.name:
            # Время установления (когда значение становится и остается ниже порога)
            threshold = self.config.threshold or 0.1  # По умолчанию 0.1 рад (~5.7 градусов)

            # Ищем последнее пересечение порога
            below_threshold = np.abs(values_array) < threshold

            # Находим индекс, после которого все значения ниже порога
            settling_idx = -1
            for i in range(len(below_threshold) - 1, -1, -1):
                if not below_threshold[i]:
                    settling_idx = i + 1 if i + 1 < len(time_steps) else len(time_steps) - 1
                    break

            if settling_idx >= 0 and settling_idx < len(time_steps):
                return float(time_steps[settling_idx])
            else:
                return float(time_steps[-1])  # Никогда не установилось

        elif 'overshoot' in self.config.name:
            # Перерегулирование (в процентах от начального отклонения)
            if len(values_array) > 1:
                initial_value = np.abs(values_array[0])
                if initial_value > 0:
                    max_deviation = np.max(np.abs(values_array))
                    overshoot = (max_deviation - initial_value) / initial_value * 100
                    return float(overshoot)

            return 0.0

        elif 'rise_time' in self.config.name:
            # Время нарастания (от 10% до 90% от начального значения)
            if len(values_array) > 1:
                initial_value = values_array[0]
                target_value = 0.0  # Целевое значение (вертикальное положение)

                # Находим индексы пересечения 10% и 90%
                ten_percent = 0.1 * initial_value
                ninety_percent = 0.9 * initial_value

                idx_10 = idx_90 = -1
                for i, val in enumerate(values_array):
                    if idx_10 < 0 and np.abs(val) <= np.abs(ten_percent):
                        idx_10 = i
                    if idx_90 < 0 and np.abs(val) <= np.abs(ninety_percent):
                        idx_90 = i

                if idx_10 >= 0 and idx_90 >= 0 and idx_10 < len(time_steps) and idx_90 < len(time_steps):
                    return float(time_steps[idx_90] - time_steps[idx_10])

            return 0.0

        else:
            raise ValueError(f"Неизвестная метрика состояния: {self.config.name}")


class CostMetric(Metric):
    """Метрики стоимости"""

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет метрику стоимости"""
        costs = data.get('costs', [])

        if not costs:
            return 0.0

        cost_array = np.array(costs)

        if self.config.name == 'avg_cost':
            return float(np.mean(cost_array))

        elif self.config.name == 'total_cost':
            return float(np.sum(cost_array))

        elif self.config.name == 'max_cost':
            return float(np.max(cost_array))

        elif self.config.name == 'min_cost':
            return float(np.min(cost_array))

        elif self.config.name == 'cost_variance':
            return float(np.var(cost_array))

        elif self.config.name == 'final_cost':
            return float(cost_array[-1]) if len(cost_array) > 0 else 0.0

        else:
            raise ValueError(f"Неизвестная метрика стоимости: {self.config.name}")


class SuccessMetric(Metric):
    """Метрики успешности выполнения"""

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет метрику успешности"""
        states = data.get('states', [])
        time_steps = data.get('time_steps', [])
        success = data.get('success', True)

        if not success:
            return 0.0

        if not states or not time_steps:
            return 0.0

        # Извлекаем углы
        angles = [s.get('theta', 0.0) for s in states]
        angles_array = np.abs(np.array(angles))

        if self.config.name == 'success':
            # Бинарная метрика успеха
            threshold = self.config.threshold or (math.pi / 4)  # 45 градусов по умолчанию
            max_angle = np.max(angles_array)
            return 1.0 if max_angle < threshold else 0.0

        elif self.config.name == 'stability_margin':
            # Запас устойчивости (насколько максимальный угол меньше порога)
            threshold = self.config.threshold or (math.pi / 4)
            max_angle = np.max(angles_array)
            if max_angle < threshold:
                return 1.0 - (max_angle / threshold)
            else:
                return 0.0

        elif self.config.name == 'time_in_target':
            # Время в целевом состоянии (угол < 5 градусов)
            target_threshold = math.radians(5.0)  # 5 градусов
            in_target = angles_array < target_threshold

            if len(time_steps) > 1:
                dt = time_steps[1] - time_steps[0]
                time_in_target = np.sum(in_target) * dt
                total_time = time_steps[-1]
                return float(time_in_target / total_time if total_time > 0 else 0.0)
            else:
                return 1.0 if in_target[0] else 0.0

        else:
            raise ValueError(f"Неизвестная метрика успешности: {self.config.name}")


class CompositeMetric(Metric):
    """Комбинированная метрика из нескольких метрик

    Паттерн: Composite - объединяет несколько метрик в одну
    """

    def __init__(self, config: MetricConfig, sub_metrics: List[Metric]):
        """Инициализирует композитную метрику

        Args:
            config: конфигурация метрики
            sub_metrics: список под-метрик
        """
        super().__init__(config)
        self.sub_metrics = sub_metrics
        self.sub_values: Dict[str, float] = {}

    def compute(self, data: Dict[str, Any]) -> float:
        """Вычисляет композитную метрику"""
        if not self.sub_metrics:
            return 0.0

        # Вычисляем все под-метрики
        total_weight = 0.0
        weighted_sum = 0.0

        for metric in self.sub_metrics:
            value = metric.compute_and_store(data)
            weight = metric.config.weight

            # Нормализуем значение если нужно
            if self.config.name == 'normalized_composite':
                # Для нормализованной композитной метрики нормализуем каждую под-метрику
                # (здесь нужны min/max значения для нормализации)
                normalized = metric.get_normalized(0.0, 1.0)  # Простая нормализация
                if normalized is not None:
                    value = normalized
                else:
                    value = 0.0

            self.sub_values[metric.config.name] = value
            weighted_sum += value * weight
            total_weight += weight

        if total_weight > 0:
            self.value = weighted_sum / total_weight
        else:
            self.value = 0.0

        self.computed = True
        return self.value

    def get_sub_metrics(self) -> Dict[str, float]:
        """Возвращает значения всех под-метрик"""
        return self.sub_values.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует метрику в словарь"""
        base_dict = super().to_dict()
        base_dict['sub_metrics'] = {
            name: metric.to_dict() for name, metric in zip(self.sub_values.keys(), self.sub_metrics)
        }
        base_dict['sub_values'] = self.sub_values
        return base_dict


class MetricCalculator:
    """Калькулятор метрик

    Паттерн: Facade - предоставляет простой интерфейс для вычисления метрик
    """

    # Предустановленные конфигурации метрик
    DEFAULT_METRICS = [
        # Метрики времени
        MetricConfig(name='avg_compute_time', metric_type=MetricType.TIME,
                     aggregation=AggregationMethod.MEAN, higher_is_better=False,
                     description='Среднее время вычисления на шаг', units='с'),
        MetricConfig(name='fps', metric_type=MetricType.TIME,
                     aggregation=AggregationMethod.MEAN,
                     description='Частота вычислений', units='кадров/с'),

        # Метрики управления
        MetricConfig(name='control_effort', metric_type=MetricType.CONTROL,
                     aggregation=AggregationMethod.INTEGRAL, higher_is_better=False,
                     description='Энергия управления (интеграл F²)', units='Н²·с'),
        MetricConfig(name='max_control', metric_type=MetricType.CONTROL,
                     aggregation=AggregationMethod.MAX, higher_is_better=False,
                     description='Максимальное управление', units='Н'),

        # Метрики состояния
        MetricConfig(name='max_angle', metric_type=MetricType.STATE,
                     aggregation=AggregationMethod.MAX, higher_is_better=False,
                     description='Максимальный угол отклонения', units='рад'),
        MetricConfig(name='settling_time', metric_type=MetricType.STATE,
                     aggregation=AggregationMethod.LAST, higher_is_better=False,
                     threshold=0.1, description='Время установления (угол < 0.1 рад)', units='с'),
        MetricConfig(name='overshoot', metric_type=MetricType.STATE,
                     aggregation=AggregationMethod.MAX, higher_is_better=False,
                     description='Перерегулирование', units='%'),

        # Метрики стоимости
        MetricConfig(name='avg_cost', metric_type=MetricType.COST,
                     aggregation=AggregationMethod.MEAN, higher_is_better=False,
                     description='Средняя стоимость на шаг', units=''),
        MetricConfig(name='total_cost', metric_type=MetricType.COST,
                     aggregation=AggregationMethod.SUM, higher_is_better=False,
                     description='Общая стоимость', units=''),

        # Метрики успешности
        MetricConfig(name='success', metric_type=MetricType.SUCCESS,
                     aggregation=AggregationMethod.LAST, threshold=math.pi / 4,
                     description='Успешность (макс угол < 45°)', units=''),
    ]

    def __init__(self, metric_configs: Optional[List[MetricConfig]] = None):
        """Инициализирует калькулятор метрик

        Args:
            metric_configs: список конфигураций метрик (если None, используются DEFAULT_METRICS)
        """
        self.metric_configs = metric_configs or self.DEFAULT_METRICS
        self.metrics: Dict[str, Metric] = {}
        self._create_metrics()

    def _create_metrics(self):
        """Создает метрики на основе конфигураций"""
        for config in self.metric_configs:
            metric = self._create_metric_from_config(config)
            self.metrics[config.name] = metric

    def _create_metric_from_config(self, config: MetricConfig) -> Metric:
        """Создает метрику на основе конфигурации"""
        if config.metric_type == MetricType.TIME:
            return TimeMetric(config)
        elif config.metric_type == MetricType.CONTROL:
            return ControlMetric(config)
        elif config.metric_type == MetricType.STATE:
            return StateMetric(config)
        elif config.metric_type == MetricType.COST:
            return CostMetric(config)
        elif config.metric_type == MetricType.SUCCESS:
            return SuccessMetric(config)
        elif config.metric_type == MetricType.COMPOSITE:
            # Для композитных метрик нужны под-метрики
            return CompositeMetric(config, [])
        else:
            raise ValueError(f"Неизвестный тип метрики: {config.metric_type}")

    def compute_all(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Вычисляет все метрики

        Args:
            data: словарь с данными для вычисления метрик

        Returns:
            словарь с вычисленными метриками
        """
        results = {}

        for name, metric in self.metrics.items():
            try:
                value = metric.compute_and_store(data)
                results[name] = value
            except Exception as e:
                warnings.warn(f"Ошибка при вычислении метрики {name}: {e}")
                results[name] = 0.0

        return results

    def compute_single(self, metric_name: str, data: Dict[str, Any]) -> float:
        """Вычисляет одну метрику

        Args:
            metric_name: имя метрики
            data: словарь с данными

        Returns:
            значение метрики
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Метрика '{metric_name}' не найдена")

        return self.metrics[metric_name].compute_and_store(data)

    def get_metric(self, metric_name: str) -> Optional[Metric]:
        """Возвращает метрику по имени"""
        return self.metrics.get(metric_name)

    def get_results(self) -> Dict[str, Optional[float]]:
        """Возвращает последние вычисленные значения всех метрик"""
        return {name: metric.get_value() for name, metric in self.metrics.items()}

    def create_composite_metric(self, name: str, sub_metric_names: List[str],
                                weights: Optional[List[float]] = None,
                                description: Optional[str] = None) -> CompositeMetric:
        """Создает композитную метрику

        Args:
            name: имя композитной метрики
            sub_metric_names: имена под-метрик
            weights: веса под-метрик (если None, все веса равны 1)
            description: описание метрики

        Returns:
            созданная композитная метрика
        """
        # Создаем под-метрики
        sub_metrics = []
        for i, sub_name in enumerate(sub_metric_names):
            if sub_name not in self.metrics:
                raise ValueError(f"Под-метрика '{sub_name}' не найдена")

            sub_metric = self.metrics[sub_name]
            # Клонируем конфигурацию с новым весом
            config = MetricConfig(
                name=sub_metric.config.name,
                metric_type=sub_metric.config.metric_type,
                aggregation=sub_metric.config.aggregation,
                weight=weights[i] if weights and i < len(weights) else 1.0,
                threshold=sub_metric.config.threshold,
                higher_is_better=sub_metric.config.higher_is_better,
                description=sub_metric.config.description,
                units=sub_metric.config.units
            )

            # Создаем новую метрику с обновленной конфигурацией
            if sub_metric.config.metric_type == MetricType.TIME:
                new_sub_metric = TimeMetric(config)
            elif sub_metric.config.metric_type == MetricType.CONTROL:
                new_sub_metric = ControlMetric(config)
            elif sub_metric.config.metric_type == MetricType.STATE:
                new_sub_metric = StateMetric(config)
            elif sub_metric.config.metric_type == MetricType.COST:
                new_sub_metric = CostMetric(config)
            elif sub_metric.config.metric_type == MetricType.SUCCESS:
                new_sub_metric = SuccessMetric(config)
            else:
                continue

            sub_metrics.append(new_sub_metric)

        # Создаем конфигурацию для композитной метрики
        composite_config = MetricConfig(
            name=name,
            metric_type=MetricType.COMPOSITE,
            description=description or f"Композитная метрика из {', '.join(sub_metric_names)}"
        )

        # Создаем композитную метрику
        composite_metric = CompositeMetric(composite_config, sub_metrics)
        self.metrics[name] = composite_metric

        return composite_metric

    def compute_performance_score(self, data: Dict[str, Any],
                                  weights: Optional[Dict[str, float]] = None) -> float:
        """Вычисляет общий балл производительности

        Args:
            data: данные для вычисления
            weights: веса метрик (если None, используются веса из конфигураций)

        Returns:
            общий балл производительности [0, 1]
        """
        # Вычисляем все метрики
        results = self.compute_all(data)

        # Нормализуем метрики
        normalized_scores = []
        total_weight = 0.0

        for name, value in results.items():
            metric = self.metrics[name]

            # Определяем min и max для нормализации
            # В реальном приложении нужно использовать исторические данные или эталонные значения
            if metric.config.higher_is_better:
                min_val = 0.0
                max_val = self._get_reference_max(name)
            else:
                min_val = self._get_reference_min(name)
                max_val = self._get_reference_max(name)

            # Нормализуем значение
            normalized = metric.get_normalized(min_val, max_val)
            if normalized is not None:
                weight = weights.get(name, 1.0) if weights else metric.config.weight
                normalized_scores.append(normalized * weight)
                total_weight += weight

        if total_weight > 0:
            return sum(normalized_scores) / total_weight
        else:
            return 0.0

    def _get_reference_min(self, metric_name: str) -> float:
        """Возвращает минимальное эталонное значение для метрики"""
        # В реальном приложении эти значения нужно брать из базы данных или конфигурации
        reference_values = {
            'avg_compute_time': 0.0,
            'fps': 0.0,
            'control_effort': 0.0,
            'max_control': 0.0,
            'max_angle': 0.0,
            'settling_time': 0.0,
            'overshoot': 0.0,
            'avg_cost': 0.0,
            'total_cost': 0.0,
            'success': 0.0
        }
        return reference_values.get(metric_name, 0.0)

    def _get_reference_max(self, metric_name: str) -> float:
        """Возвращает максимальное эталонное значение для метрики"""
        reference_values = {
            'avg_compute_time': 0.1,  # 100 мс
            'fps': 1000.0,  # 1000 кадров/с
            'control_effort': 1000.0,  # Большое усилие управления
            'max_control': 20.0,  # 20 Н
            'max_angle': math.pi,  # 180 градусов
            'settling_time': 10.0,  # 10 секунд
            'overshoot': 100.0,  # 100%
            'avg_cost': 1000.0,  # Большая стоимость
            'total_cost': 10000.0,  # Очень большая общая стоимость
            'success': 1.0  # Бинарная метрика
        }
        return reference_values.get(metric_name, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует калькулятор в словарь"""
        return {
            'metrics': {name: metric.to_dict() for name, metric in self.metrics.items()},
            'configs': [config.to_dict() for config in self.metric_configs]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricCalculator':
        """Создает калькулятор из словаря"""
        # Восстанавливаем конфигурации
        configs = [MetricConfig.from_dict(c) for c in data['configs']]

        # Создаем калькулятор
        calculator = cls(configs)

        # Восстанавливаем значения метрик если есть
        if 'metrics' in data:
            for name, metric_data in data['metrics'].items():
                if name in calculator.metrics:
                    calculator.metrics[name].value = metric_data.get('value')
                    calculator.metrics[name].computed = metric_data.get('computed', False)

        return calculator

    def save(self, filename: str):
        """Сохраняет калькулятор в файл"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filename: str) -> 'MetricCalculator':
        """Загружает калькулятор из файла"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Функции высокого уровня

def calculate_all_metrics(data: Dict[str, Any],
                          configs: Optional[List[MetricConfig]] = None) -> Dict[str, Any]:
    """Вычисляет все метрики для данных

    Args:
        data: словарь с данными
        configs: конфигурации метрик (если None, используются стандартные)

    Returns:
        словарь с метриками и дополнительной информацией
    """
    calculator = MetricCalculator(configs)
    metrics = calculator.compute_all(data)

    # Вычисляем общий балл производительности
    performance_score = calculator.compute_performance_score(data)

    return {
        'metrics': metrics,
        'performance_score': performance_score,
        'calculator': calculator.to_dict()
    }


def compare_metrics(results_list: List[Dict[str, Any]],
                    metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Сравнивает метрики нескольких результатов

    Args:
        results_list: список результатов с метриками
        metric_names: имена метрик для сравнения (если None, все метрики)

    Returns:
        словарь с результатами сравнения
    """
    if not results_list:
        return {}

    # Собираем все метрики
    all_metrics = set()
    for result in results_list:
        if 'metrics' in result:
            all_metrics.update(result['metrics'].keys())

    # Фильтруем метрики если указаны
    if metric_names:
        metrics_to_compare = [m for m in metric_names if m in all_metrics]
    else:
        metrics_to_compare = list(all_metrics)

    # Сравниваем метрики
    comparison = {
        'metrics': {},
        'rankings': {},
        'summary': {}
    }

    for metric in metrics_to_compare:
        values = []
        for i, result in enumerate(results_list):
            if 'metrics' in result and metric in result['metrics']:
                values.append((i, result['metrics'][metric]))

        if values:
            # Сортируем по значению (лучшие первые)
            # Определяем лучше ли большее значение
            # (это упрощение, в реальном приложении нужно учитывать higher_is_better)
            values.sort(key=lambda x: x[1], reverse=True)

            comparison['metrics'][metric] = {
                'values': {f'result_{i}': val for i, val in values},
                'best': values[0][0],
                'worst': values[-1][0],
                'mean': np.mean([v for _, v in values]),
                'std': np.std([v for _, v in values])
            }

    # Ранжируем результаты по общему баллу
    if all('performance_score' in r for r in results_list):
        scores = [(i, r['performance_score']) for i, r in enumerate(results_list)]
        scores.sort(key=lambda x: x[1], reverse=True)

        comparison['rankings'] = {
            f'rank_{j + 1}': i for j, (i, _) in enumerate(scores)
        }

        comparison['summary'] = {
            'best_overall': scores[0][0],
            'worst_overall': scores[-1][0],
            'avg_score': np.mean([s for _, s in scores]),
            'score_std': np.std([s for _, s in scores])
        }

    return comparison


def create_metric_summary(metrics: Dict[str, float],
                          calculator: Optional[MetricCalculator] = None) -> Dict[str, Any]:
    """Создает сводку по метрикам

    Args:
        metrics: словарь с метриками
        calculator: калькулятор метрик для дополнительной информации

    Returns:
        сводка по метрикам
    """
    summary = {
        'basic': {},
        'by_category': {},
        'recommendations': []
    }

    # Категории метрик
    categories = {
        'Производительность': ['avg_compute_time', 'fps', 'total_compute_time'],
        'Управление': ['control_effort', 'max_control', 'control_smoothness'],
        'Качество': ['max_angle', 'settling_time', 'overshoot', 'success'],
        'Эффективность': ['avg_cost', 'total_cost', 'final_cost']
    }

    # Базовые метрики
    for metric_name in ['success', 'avg_compute_time', 'max_angle', 'control_effort']:
        if metric_name in metrics:
            summary['basic'][metric_name] = metrics[metric_name]

    # Метрики по категориям
    for category, metric_names in categories.items():
        category_metrics = {}
        for metric_name in metric_names:
            if metric_name in metrics:
                category_metrics[metric_name] = metrics[metric_name]

        if category_metrics:
            summary['by_category'][category] = category_metrics

    # Рекомендации
    if 'success' in metrics and metrics['success'] == 0:
        summary['recommendations'].append(
            "❌ Алгоритм не смог стабилизировать маятник. "
            "Попробуйте увеличить горизонт планирования или количество траекторий."
        )

    if 'avg_compute_time' in metrics and metrics['avg_compute_time'] > 0.05:
        summary['recommendations'].append(
            "⚠️ Время вычисления высокое. "
            "Рассмотрите оптимизацию алгоритма или использование другой реализации (JAX/C++)."
        )

    if 'max_angle' in metrics and metrics['max_angle'] > math.radians(30):
        summary['recommendations'].append(
            "⚠️ Большое отклонение маятника. "
            "Увеличьте веса в функции стоимости для угла и угловой скорости."
        )

    if 'control_effort' in metrics and metrics['control_effort'] > 500:
        summary['recommendations'].append(
            "⚠️ Высокое усилие управления. "
            "Увеличьте вес для штрафа управления в функции стоимости."
        )

    # Общая оценка
    if 'success' in metrics and metrics['success'] == 1:
        if 'settling_time' in metrics and metrics['settling_time'] < 2.0:
            summary['overall_rating'] = "Отлично ⭐⭐⭐⭐⭐"
        else:
            summary['overall_rating'] = "Хорошо ⭐⭐⭐⭐"
    else:
        summary['overall_rating'] = "Требует улучшения ⭐⭐"

    return summary


def save_metrics_report(metrics: Dict[str, Any], filename: str):
    """Сохраняет отчет по метрикам в файл"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics.get('metrics', {}),
        'performance_score': metrics.get('performance_score', 0.0),
        'summary': create_metric_summary(metrics.get('metrics', {}))
    }

    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Отчет по метрикам сохранен в {filename}")


# Декораторы для измерения производительности

def time_metric(func):
    """Декоратор для измерения времени выполнения функции"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time

        # Сохраняем время выполнения если в аргументах есть collector
        for arg in args:
            if isinstance(arg, ResultsCollector):
                # Создаем запись метрики времени
                metric_data = {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'timestamp': datetime.now().isoformat()
                }
                # Можно сохранить в коллектор
                break

        print(f"Функция {func.__name__} выполнена за {execution_time:.3f} секунд")

        return result

    return wrapper


def track_metrics(metric_names: List[str]):
    """Декоратор для отслеживания метрик во время выполнения"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Здесь можно добавить логику отслеживания метрик
            print(f"Отслеживание метрик: {', '.join(metric_names)}")
            return func(*args, **kwargs)

        return wrapper

    return decorator