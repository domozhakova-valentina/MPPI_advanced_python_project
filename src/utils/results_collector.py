"""
Сборщик результатов для агрегации, хранения и анализа данных экспериментов.

Паттерны:
- Repository: централизованное хранилище результатов
- Builder: построение сложных запросов к данным
- Iterator: итерация по результатам
- Memento: сохранение и восстановление состояния
"""

import json
import pickle
import csv
import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union, Iterator, Callable
from datetime import datetime
from pathlib import Path
import hashlib
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import warnings
from contextlib import contextmanager
import zipfile
import yaml


class DataFormat(Enum):
    """Форматы данных для экспорта"""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PICKLE = "pickle"
    PARQUET = "parquet"
    YAML = "yaml"


@dataclass
class ResultEntry:
    """Запись результата эксперимента

    Паттерн: Value Object
    """
    # Идентификационные данные
    experiment_id: str  # Уникальный ID эксперимента
    run_id: str  # Уникальный ID запуска
    timestamp: datetime  # Время создания записи

    # Конфигурации
    implementation: str  # Реализация (numpy, jax, cpp)
    system_config: Dict[str, Any]  # Конфигурация системы
    mppi_config: Dict[str, Any]  # Конфигурация MPPI
    simulation_config: Dict[str, Any]  # Конфигурация симуляции

    # Данные
    states: List[Dict[str, float]]  # Состояния системы
    controls: List[float]  # Управления
    costs: List[float]  # Значения стоимости
    compute_times: List[float]  # Время вычислений
    time_steps: List[float]  # Временные шаги

    # Метаданные
    success: bool = True  # Успешность выполнения
    error_message: Optional[str] = None  # Сообщение об ошибке (если есть)
    tags: List[str] = field(default_factory=list)  # Теги для категоризации
    notes: Optional[str] = None  # Дополнительные заметки

    # Метрики (опционально, могут быть вычислены позже)
    metrics: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Пост-инициализация"""
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultEntry':
        """Создает из словаря"""
        return cls(**data)

    def get_hash(self) -> str:
        """Вычисляет хэш записи для уникальной идентификации"""
        # Используем только ключевые поля для хэширования
        key_data = {
            'experiment_id': self.experiment_id,
            'run_id': self.run_id,
            'implementation': self.implementation,
            'timestamp': self.timestamp.isoformat()
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_duration(self) -> float:
        """Возвращает продолжительность симуляции"""
        if self.time_steps:
            return self.time_steps[-1]
        return 0.0

    def get_num_steps(self) -> int:
        """Возвращает количество шагов"""
        return len(self.time_steps)

    def get_final_state(self) -> Optional[Dict[str, float]]:
        """Возвращает конечное состояние"""
        if self.states:
            return self.states[-1]
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает базовую статистику"""
        if not self.controls:
            return {}

        controls_array = np.array(self.controls)
        costs_array = np.array(self.costs)
        compute_times_array = np.array(self.compute_times)

        return {
            'control_stats': {
                'mean': float(np.mean(controls_array)),
                'std': float(np.std(controls_array)),
                'min': float(np.min(controls_array)),
                'max': float(np.max(controls_array)),
                'abs_mean': float(np.mean(np.abs(controls_array)))
            },
            'cost_stats': {
                'mean': float(np.mean(costs_array)),
                'std': float(np.std(costs_array)),
                'min': float(np.min(costs_array)),
                'max': float(np.max(costs_array)),
                'total': float(np.sum(costs_array))
            },
            'time_stats': {
                'mean': float(np.mean(compute_times_array)),
                'std': float(np.std(compute_times_array)),
                'min': float(np.min(compute_times_array)),
                'max': float(np.max(compute_times_array)),
                'total': float(np.sum(compute_times_array))
            }
        }


@dataclass
class ExperimentRun:
    """Запуск эксперимента с несколькими результатами

    Паттерн: Composite - агрегирует несколько ResultEntry
    """
    experiment_name: str  # Название эксперимента
    description: Optional[str] = None  # Описание
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    results: List[ResultEntry] = field(default_factory=list)  # Результаты

    def add_result(self, result: ResultEntry):
        """Добавляет результат в эксперимент"""
        self.results.append(result)
        self.updated_at = datetime.now()

    def remove_result(self, result_hash: str):
        """Удаляет результат по хэшу"""
        self.results = [r for r in self.results if r.get_hash() != result_hash]
        self.updated_at = datetime.now()

    def get_result(self, result_hash: str) -> Optional[ResultEntry]:
        """Возвращает результат по хэшу"""
        for result in self.results:
            if result.get_hash() == result_hash:
                return result
        return None

    def filter_results(self, condition: Callable[[ResultEntry], bool]) -> List[ResultEntry]:
        """Фильтрует результаты по условию"""
        return [r for r in self.results if condition(r)]

    def get_implementations(self) -> List[str]:
        """Возвращает список уникальных реализаций"""
        return list(set(r.implementation for r in self.results))

    def get_success_rate(self) -> float:
        """Возвращает процент успешных запусков"""
        if not self.results:
            return 0.0
        successful = sum(1 for r in self.results if r.success)
        return successful / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует в словарь"""
        return {
            'experiment_name': self.experiment_name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'num_results': len(self.results),
            'results': [r.to_dict() for r in self.results]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Создает из словаря"""
        experiment = cls(
            experiment_name=data['experiment_name'],
            description=data.get('description'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )

        # Восстанавливаем результаты
        for result_data in data['results']:
            result = ResultEntry.from_dict(result_data)
            experiment.add_result(result)

        return experiment


class ResultsCollector:
    """Коллектор результатов экспериментов

    Паттерн: Repository - централизованное хранилище с методами доступа
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Инициализирует коллектор

        Args:
            storage_path: путь для сохранения данных (опционально)
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.experiments: Dict[str, ExperimentRun] = {}
        self._next_id = 1

        # Создаем директорию для хранения, если указан путь
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        # Кэш для быстрого доступа
        self._result_cache: Dict[str, ResultEntry] = {}
        self._index: Dict[str, List[str]] = {
            'by_implementation': {},
            'by_experiment': {},
            'by_tag': {},
            'by_success': {'success': [], 'failed': []}
        }

        # Загружаем существующие данные
        if self.storage_path and self.storage_path.exists():
            self._load_from_storage()

    def _load_from_storage(self):
        """Загружает данные из хранилища"""
        try:
            metadata_file = self.storage_path / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                for exp_name, exp_file in metadata['experiments'].items():
                    exp_path = self.storage_path / exp_file
                    if exp_path.exists():
                        with open(exp_path, 'r') as f:
                            exp_data = json.load(f)
                        experiment = ExperimentRun.from_dict(exp_data)
                        self.experiments[exp_name] = experiment

            print(f"Загружено {len(self.experiments)} экспериментов из хранилища")
        except Exception as e:
            print(f"Ошибка при загрузке данных: {e}")

    def _save_to_storage(self):
        """Сохраняет данные в хранилище"""
        if not self.storage_path:
            return

        try:
            metadata = {'experiments': {}}

            for exp_name, experiment in self.experiments.items():
                exp_file = f"experiment_{exp_name}.json"
                exp_path = self.storage_path / exp_file

                # Сохраняем эксперимент
                with open(exp_path, 'w') as f:
                    json.dump(experiment.to_dict(), f, indent=2)

                metadata['experiments'][exp_name] = exp_file

            # Сохраняем метаданные
            metadata_file = self.storage_path / 'metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            print(f"Ошибка при сохранении данных: {e}")

    def _update_index(self, result: ResultEntry):
        """Обновляет индексы для быстрого поиска"""
        result_hash = result.get_hash()

        # По реализации
        impl = result.implementation
        if impl not in self._index['by_implementation']:
            self._index['by_implementation'][impl] = []
        self._index['by_implementation'][impl].append(result_hash)

        # По эксперименту
        exp_id = result.experiment_id
        if exp_id not in self._index['by_experiment']:
            self._index['by_experiment'][exp_id] = []
        self._index['by_experiment'][exp_id].append(result_hash)

        # По тегам
        for tag in result.tags:
            if tag not in self._index['by_tag']:
                self._index['by_tag'][tag] = []
            self._index['by_tag'][tag].append(result_hash)

        # По успешности
        if result.success:
            self._index['by_success']['success'].append(result_hash)
        else:
            self._index['by_success']['failed'].append(result_hash)

        # Кэшируем результат
        self._result_cache[result_hash] = result

    def create_experiment(self, name: str, description: Optional[str] = None) -> str:
        """Создает новый эксперимент

        Args:
            name: название эксперимента
            description: описание эксперимента

        Returns:
            ID созданного эксперимента
        """
        if name in self.experiments:
            raise ValueError(f"Эксперимент с именем '{name}' уже существует")

        experiment = ExperimentRun(
            experiment_name=name,
            description=description
        )

        self.experiments[name] = experiment
        return name

    def add_result(self, experiment_name: str, result: ResultEntry) -> str:
        """Добавляет результат в эксперимент

        Args:
            experiment_name: название эксперимента
            result: результат для добавления

        Returns:
            хэш добавленного результата
        """
        if experiment_name not in self.experiments:
            # Создаем эксперимент, если он не существует
            self.create_experiment(experiment_name)

        experiment = self.experiments[experiment_name]

        # Генерируем уникальный run_id если не указан
        if not result.run_id:
            result.run_id = f"run_{self._next_id:06d}"
            self._next_id += 1

        # Устанавливаем experiment_id если не указан
        if not result.experiment_id:
            result.experiment_id = experiment_name

        # Добавляем результат
        experiment.add_result(result)

        # Обновляем индексы
        self._update_index(result)

        # Сохраняем в хранилище
        self._save_to_storage()

        return result.get_hash()

    def add_raw_data(self, experiment_name: str, implementation: str,
                     system_config: Dict[str, Any], mppi_config: Dict[str, Any],
                     simulation_config: Dict[str, Any], states: List[Dict[str, float]],
                     controls: List[float], costs: List[float],
                     compute_times: List[float], time_steps: List[float],
                     success: bool = True, error_message: Optional[str] = None,
                     tags: Optional[List[str]] = None, notes: Optional[str] = None) -> str:
        """Добавляет сырые данные как результат

        Args:
            experiment_name: название эксперимента
            implementation: реализация (numpy, jax, cpp)
            system_config: конфигурация системы
            mppi_config: конфигурация MPPI
            simulation_config: конфигурация симуляции
            states: состояния системы
            controls: управления
            costs: значения стоимости
            compute_times: время вычислений
            time_steps: временные шаги
            success: успешность выполнения
            error_message: сообщение об ошибке
            tags: теги для категоризации
            notes: дополнительные заметки

        Returns:
            хэш добавленного результата
        """
        # Создаем запись результата
        result = ResultEntry(
            experiment_id=experiment_name,
            run_id=f"raw_{len(self.get_all_results()) + 1:06d}",
            timestamp=datetime.now(),
            implementation=implementation,
            system_config=system_config,
            mppi_config=mppi_config,
            simulation_config=simulation_config,
            states=states,
            controls=controls,
            costs=costs,
            compute_times=compute_times,
            time_steps=time_steps,
            success=success,
            error_message=error_message,
            tags=tags or [],
            notes=notes
        )

        return self.add_result(experiment_name, result)

    def get_result(self, result_hash: str) -> Optional[ResultEntry]:
        """Возвращает результат по хэшу

        Args:
            result_hash: хэш результата

        Returns:
            результат или None, если не найден
        """
        # Пробуем получить из кэша
        if result_hash in self._result_cache:
            return self._result_cache[result_hash]

        # Ищем во всех экспериментах
        for experiment in self.experiments.values():
            result = experiment.get_result(result_hash)
            if result:
                self._result_cache[result_hash] = result
                return result

        return None

    def get_experiment(self, experiment_name: str) -> Optional[ExperimentRun]:
        """Возвращает эксперимент по имени

        Args:
            experiment_name: название эксперимента

        Returns:
            эксперимент или None, если не найден
        """
        return self.experiments.get(experiment_name)

    def get_all_results(self) -> List[ResultEntry]:
        """Возвращает все результаты всех экспериментов"""
        all_results = []
        for experiment in self.experiments.values():
            all_results.extend(experiment.results)
        return all_results

    def filter_results(self, **criteria) -> List[ResultEntry]:
        """Фильтрует результаты по критериям

        Args:
            **criteria: критерии фильтрации:
                - implementation: фильтр по реализации
                - experiment_id: фильтр по ID эксперимента
                - success: фильтр по успешности
                - tag: фильтр по тегу
                - min_steps: минимальное количество шагов
                - max_steps: максимальное количество шагов
                - start_date: начальная дата
                - end_date: конечная дата

        Returns:
            отфильтрованные результаты
        """
        results = self.get_all_results()

        for key, value in criteria.items():
            if value is None:
                continue

            if key == 'implementation':
                results = [r for r in results if r.implementation == value]
            elif key == 'experiment_id':
                results = [r for r in results if r.experiment_id == value]
            elif key == 'success':
                results = [r for r in results if r.success == value]
            elif key == 'tag':
                results = [r for r in results if value in r.tags]
            elif key == 'min_steps':
                results = [r for r in results if len(r.time_steps) >= value]
            elif key == 'max_steps':
                results = [r for r in results if len(r.time_steps) <= value]
            elif key == 'start_date':
                start_date = value if isinstance(value, datetime) else datetime.fromisoformat(value)
                results = [r for r in results if r.timestamp >= start_date]
            elif key == 'end_date':
                end_date = value if isinstance(value, datetime) else datetime.fromisoformat(value)
                results = [r for r in results if r.timestamp <= end_date]

        return results

    def query(self, query_string: str) -> List[ResultEntry]:
        """Выполняет запрос к результатам

        Args:
            query_string: строка запроса в формате "field operator value"
                         Примеры: "implementation == numpy", "success == True",
                                 "len(time_steps) > 100"

        Returns:
            результаты, удовлетворяющие запросу
        """
        # Простая реализация запросов
        # В реальном приложении можно использовать более сложный парсер

        try:
            field, operator, value_str = query_string.split()

            # Преобразуем значение
            if value_str.lower() == 'true':
                value = True
            elif value_str.lower() == 'false':
                value = False
            elif value_str.isdigit():
                value = int(value_str)
            elif value_str.replace('.', '', 1).isdigit():
                value = float(value_str)
            else:
                value = value_str.strip('"\'')

            results = self.get_all_results()
            filtered = []

            for result in results:
                # Получаем значение поля
                if field == 'len(time_steps)':
                    field_value = len(result.time_steps)
                else:
                    field_value = getattr(result, field, None)

                # Применяем оператор
                if operator == '==':
                    if field_value == value:
                        filtered.append(result)
                elif operator == '!=':
                    if field_value != value:
                        filtered.append(result)
                elif operator == '>':
                    if field_value > value:
                        filtered.append(result)
                elif operator == '>=':
                    if field_value >= value:
                        filtered.append(result)
                elif operator == '<':
                    if field_value < value:
                        filtered.append(result)
                elif operator == '<=':
                    if field_value <= value:
                        filtered.append(result)
                elif operator == 'in':
                    if value in field_value:
                        filtered.append(result)

            return filtered

        except Exception as e:
            print(f"Ошибка выполнения запроса: {e}")
            return []

    def delete_result(self, result_hash: str) -> bool:
        """Удаляет результат

        Args:
            result_hash: хэш результата

        Returns:
            True если удалено, False если не найдено
        """
        for experiment in self.experiments.values():
            for i, result in enumerate(experiment.results):
                if result.get_hash() == result_hash:
                    # Удаляем из эксперимента
                    experiment.results.pop(i)

                    # Удаляем из индексов
                    self._remove_from_index(result_hash)

                    # Удаляем из кэша
                    if result_hash in self._result_cache:
                        del self._result_cache[result_hash]

                    # Сохраняем изменения
                    self._save_to_storage()

                    return True

        return False

    def _remove_from_index(self, result_hash: str):
        """Удаляет результат из всех индексов"""
        for index_type in self._index.values():
            if isinstance(index_type, dict):
                for key, hashes in index_type.items():
                    if result_hash in hashes:
                        index_type[key].remove(result_hash)
            elif isinstance(index_type, list):
                if result_hash in index_type:
                    index_type.remove(result_hash)

    def export_experiment(self, experiment_name: str,
                          export_format: DataFormat = DataFormat.JSON,
                          output_path: Optional[str] = None) -> str:
        """Экспортирует эксперимент в файл

        Args:
            experiment_name: название эксперимента
            export_format: формат экспорта
            output_path: путь для сохранения (если None, генерируется автоматически)

        Returns:
            путь к экспортированному файлу
        """
        if experiment_name not in self.experiments:
            raise ValueError(f"Эксперимент '{experiment_name}' не найден")

        experiment = self.experiments[experiment_name]

        # Генерируем имя файла если не указано
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{experiment_name}_{timestamp}.{export_format.value}"
            output_path = filename
        else:
            output_path = Path(output_path)

        # Экспортируем в выбранном формате
        if export_format == DataFormat.JSON:
            with open(output_path, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2)

        elif export_format == DataFormat.CSV:
            # Экспортируем как несколько CSV файлов
            self._export_experiment_to_csv(experiment, output_path)

        elif export_format == DataFormat.EXCEL:
            self._export_experiment_to_excel(experiment, output_path)

        elif export_format == DataFormat.PICKLE:
            with open(output_path, 'wb') as f:
                pickle.dump(experiment, f)

        elif export_format == DataFormat.YAML:
            with open(output_path, 'w') as f:
                yaml.dump(experiment.to_dict(), f)

        else:
            raise ValueError(f"Неподдерживаемый формат: {export_format}")

        return str(output_path)

    def _export_experiment_to_csv(self, experiment: ExperimentRun, output_path: str):
        """Экспортирует эксперимент в CSV"""
        output_path = Path(output_path)

        # Создаем директорию если нужно
        if not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)

        # Экспортируем метаданные эксперимента
        metadata_file = output_path / f"{experiment.experiment_name}_metadata.csv"
        metadata_df = pd.DataFrame([{
            'experiment_name': experiment.experiment_name,
            'description': experiment.description,
            'created_at': experiment.created_at,
            'updated_at': experiment.updated_at,
            'num_results': len(experiment.results),
            'success_rate': experiment.get_success_rate()
        }])
        metadata_df.to_csv(metadata_file, index=False)

        # Экспортируем каждый результат
        for i, result in enumerate(experiment.results):
            result_file = output_path / f"result_{i:04d}.csv"

            # Создаем DataFrame с данными результата
            data = {
                'time_step': result.time_steps,
                'control': result.controls,
                'cost': result.costs,
                'compute_time': result.compute_times
            }

            # Добавляем состояния
            for j, key in enumerate(['x', 'theta', 'x_dot', 'theta_dot']):
                values = [state.get(key, 0.0) for state in result.states]
                data[key] = values

            df = pd.DataFrame(data)
            df.to_csv(result_file, index=False)

    def _export_experiment_to_excel(self, experiment: ExperimentRun, output_path: str):
        """Экспортирует эксперимент в Excel"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Лист с метаданными
            metadata_df = pd.DataFrame([{
                'experiment_name': experiment.experiment_name,
                'description': experiment.description,
                'created_at': experiment.created_at,
                'updated_at': experiment.updated_at,
                'num_results': len(experiment.results),
                'success_rate': experiment.get_success_rate()
            }])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)

            # Лист с сводкой результатов
            summary_data = []
            for result in experiment.results:
                stats = result.get_statistics()
                summary_data.append({
                    'run_id': result.run_id,
                    'implementation': result.implementation,
                    'timestamp': result.timestamp,
                    'num_steps': len(result.time_steps),
                    'duration': result.get_duration(),
                    'success': result.success,
                    'avg_control': stats.get('control_stats', {}).get('abs_mean', 0),
                    'total_cost': stats.get('cost_stats', {}).get('total', 0),
                    'avg_compute_time': stats.get('time_stats', {}).get('mean', 0)
                })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Листы с данными каждого результата
            for i, result in enumerate(experiment.results):
                sheet_name = f"Result_{i:03d}"

                data = {
                    'time_step': result.time_steps,
                    'control': result.controls,
                    'cost': result.costs,
                    'compute_time': result.compute_times
                }

                for j, key in enumerate(['x', 'theta', 'x_dot', 'theta_dot']):
                    values = [state.get(key, 0.0) for state in result.states]
                    data[key] = values

                df = pd.DataFrame(data)
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def import_experiment(self, file_path: str,
                          import_format: Optional[DataFormat] = None) -> str:
        """Импортирует эксперимент из файла

        Args:
            file_path: путь к файлу
            import_format: формат файла (если None, определяется по расширению)

        Returns:
            название импортированного эксперимента
        """
        file_path = Path(file_path)

        # Определяем формат если не указан
        if import_format is None:
            ext = file_path.suffix.lower()[1:]  # Без точки
            try:
                import_format = DataFormat(ext)
            except ValueError:
                raise ValueError(f"Неизвестный формат файла: {ext}")

        # Импортируем в зависимости от формата
        if import_format == DataFormat.JSON:
            with open(file_path, 'r') as f:
                data = json.load(f)

            experiment = ExperimentRun.from_dict(data)

        elif import_format == DataFormat.PICKLE:
            with open(file_path, 'rb') as f:
                experiment = pickle.load(f)

        elif import_format == DataFormat.YAML:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)

            experiment = ExperimentRun.from_dict(data)

        else:
            raise ValueError(f"Неподдерживаемый формат для импорта: {import_format}")

        # Добавляем эксперимент в коллектор
        # Если эксперимент с таким именем уже существует, добавляем суффикс
        base_name = experiment.experiment_name
        new_name = base_name
        counter = 1

        while new_name in self.experiments:
            new_name = f"{base_name}_imported_{counter}"
            counter += 1

        experiment.experiment_name = new_name
        self.experiments[new_name] = experiment

        # Индексируем результаты
        for result in experiment.results:
            self._update_index(result)

        # Сохраняем изменения
        self._save_to_storage()

        return new_name

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику по всем данным"""
        all_results = self.get_all_results()

        if not all_results:
            return {}

        implementations = {}
        experiments = {}
        success_stats = {'success': 0, 'failed': 0, 'total': len(all_results)}

        for result in all_results:
            # Статистика по реализациям
            impl = result.implementation
            if impl not in implementations:
                implementations[impl] = {'count': 0, 'success': 0, 'total_steps': 0}
            implementations[impl]['count'] += 1
            implementations[impl]['success'] += 1 if result.success else 0
            implementations[impl]['total_steps'] += len(result.time_steps)

            # Статистика по экспериментам
            exp_id = result.experiment_id
            if exp_id not in experiments:
                experiments[exp_id] = {'count': 0, 'success': 0}
            experiments[exp_id]['count'] += 1
            experiments[exp_id]['success'] += 1 if result.success else 0

            # Общая статистика успешности
            if result.success:
                success_stats['success'] += 1
            else:
                success_stats['failed'] += 1

        # Вычисляем проценты
        for impl in implementations:
            total = implementations[impl]['count']
            implementations[impl]['success_rate'] = implementations[impl]['success'] / total if total > 0 else 0
            implementations[impl]['avg_steps'] = implementations[impl]['total_steps'] / total if total > 0 else 0

        for exp in experiments:
            total = experiments[exp]['count']
            experiments[exp]['success_rate'] = experiments[exp]['success'] / total if total > 0 else 0

        success_stats['success_rate'] = success_stats['success'] / success_stats['total'] if success_stats[
                                                                                                 'total'] > 0 else 0

        return {
            'total_results': len(all_results),
            'total_experiments': len(self.experiments),
            'success_stats': success_stats,
            'implementations': implementations,
            'experiments': experiments,
            'timeline': {
                'first_result': min(r.timestamp for r in all_results).isoformat(),
                'last_result': max(r.timestamp for r in all_results).isoformat()
            }
        }

    def clear(self):
        """Очищает все данные"""
        self.experiments.clear()
        self._result_cache.clear()
        self._index = {
            'by_implementation': {},
            'by_experiment': {},
            'by_tag': {},
            'by_success': {'success': [], 'failed': []}
        }

        # Удаляем файлы хранилища если есть
        if self.storage_path and self.storage_path.exists():
            for file in self.storage_path.glob("*.json"):
                file.unlink()
            metadata_file = self.storage_path / 'metadata.json'
            if metadata_file.exists():
                metadata_file.unlink()

    def __str__(self) -> str:
        """Строковое представление"""
        stats = self.get_statistics()
        return (f"ResultsCollector(\n"
                f"  experiments: {len(self.experiments)},\n"
                f"  total_results: {stats.get('total_results', 0)},\n"
                f"  success_rate: {stats.get('success_stats', {}).get('success_rate', 0):.1%},\n"
                f"  storage: {self.storage_path}\n"
                f")")

    def __len__(self) -> int:
        """Возвращает количество результатов"""
        return len(self.get_all_results())


# Декораторы для работы с коллектором

def with_collector(func):
    """Декоратор для автоматического создания коллектора

    Паттерн: Decorator
    """

    def wrapper(*args, **kwargs):
        collector = ResultsCollector()
        return func(collector, *args, **kwargs)

    return wrapper


def save_on_exit(func):
    """Декоратор для автоматического сохранения коллектора при выходе"""

    def wrapper(collector: ResultsCollector, *args, **kwargs):
        try:
            result = func(collector, *args, **kwargs)
            collector._save_to_storage()
            return result
        except Exception as e:
            print(f"Ошибка в функции {func.__name__}: {e}")
            raise

    return wrapper


# Функции высокого уровня

def save_results(results: List[ResultEntry], filename: str,
                 format: DataFormat = DataFormat.JSON):
    """Сохраняет список результатов в файл"""
    collector = ResultsCollector()
    experiment_name = "exported_results"

    for result in results:
        collector.add_result(experiment_name, result)

    collector.export_experiment(experiment_name, format, filename)


def load_results(filename: str, format: Optional[DataFormat] = None) -> List[ResultEntry]:
    """Загружает результаты из файла"""
    collector = ResultsCollector()
    experiment_name = collector.import_experiment(filename, format)
    experiment = collector.get_experiment(experiment_name)

    if experiment:
        return experiment.results
    return []


def merge_results(collectors: List[ResultsCollector]) -> ResultsCollector:
    """Объединяет несколько коллекторов в один"""
    merged = ResultsCollector()

    for collector in collectors:
        for experiment_name, experiment in collector.experiments.items():
            # Добавляем эксперимент если его еще нет
            if experiment_name not in merged.experiments:
                merged.experiments[experiment_name] = experiment
            else:
                # Или добавляем результаты к существующему эксперименту
                for result in experiment.results:
                    merged.add_result(experiment_name, result)

    return merged


def filter_results(collector: ResultsCollector, **criteria) -> ResultsCollector:
    """Создает новый коллектор с отфильтрованными результатами"""
    filtered = ResultsCollector()
    filtered_results = collector.filter_results(**criteria)

    for result in filtered_results:
        filtered.add_result(result.experiment_id, result)

    return filtered


def export_to_csv(collector: ResultsCollector, output_dir: str):
    """Экспортирует все эксперименты в CSV формат"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment_name in collector.experiments:
        collector.export_experiment(experiment_name, DataFormat.CSV,
                                    str(output_dir / experiment_name))


def export_to_excel(collector: ResultsCollector, filename: str):
    """Экспортирует все эксперименты в один Excel файл"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for i, (experiment_name, experiment) in enumerate(collector.experiments.items()):
            # Создаем сводку по эксперименту
            summary_data = []
            for result in experiment.results:
                stats = result.get_statistics()
                summary_data.append({
                    'run_id': result.run_id,
                    'implementation': result.implementation,
                    'success': result.success,
                    'num_steps': len(result.time_steps),
                    'duration': result.get_duration(),
                    'avg_compute_time': stats.get('time_stats', {}).get('mean', 0),
                    'total_cost': stats.get('cost_stats', {}).get('total', 0)
                })

            if summary_data:
                df = pd.DataFrame(summary_data)
                # Ограничиваем длину имени листа
                sheet_name = experiment_name[:31]  # Excel ограничение
                if i > 0:
                    sheet_name = f"{sheet_name}_{i}"
                df.to_excel(writer, sheet_name=sheet_name, index=False)


def plot_comparison(collector: ResultsCollector, experiment_name: str,
                    save_path: Optional[str] = None):
    """Создает графики сравнения для эксперимента"""
    if experiment_name not in collector.experiments:
        raise ValueError(f"Эксперимент '{experiment_name}' не найден")

    experiment = collector.experiments[experiment_name]

    # Настраиваем стиль графиков
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Сравнение результатов: {experiment_name}', fontsize=16)

    colors = {'numpy': 'blue', 'jax': 'red', 'cpp': 'green'}
    markers = {'numpy': 'o', 'jax': 's', 'cpp': '^'}

    # Группируем результаты по реализации
    results_by_impl = {}
    for result in experiment.results:
        if result.implementation not in results_by_impl:
            results_by_impl[result.implementation] = []
        results_by_impl[result.implementation].append(result)

    # График 1: Угол отклонения во времени
    ax = axes[0, 0]
    for impl, results in results_by_impl.items():
        if results:
            # Берем первый результат для примера
            result = results[0]
            time_steps = result.time_steps
            angles = [state['theta'] for state in result.states]
            ax.plot(time_steps, angles, label=impl, color=colors.get(impl, 'gray'),
                    marker=markers.get(impl, 'o'), markersize=4, linewidth=2)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Угол (рад)')
    ax.set_title('Угол отклонения маятника')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 2: Управление во времени
    ax = axes[0, 1]
    for impl, results in results_by_impl.items():
        if results:
            result = results[0]
            time_steps = result.time_steps
            controls = result.controls
            ax.plot(time_steps, controls, label=impl, color=colors.get(impl, 'gray'),
                    marker=markers.get(impl, 'o'), markersize=4, linewidth=2)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Управление (Н)')
    ax.set_title('Приложенная сила')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 3: Стоимость во времени
    ax = axes[1, 0]
    for impl, results in results_by_impl.items():
        if results:
            result = results[0]
            time_steps = result.time_steps
            costs = result.costs
            ax.plot(time_steps, costs, label=impl, color=colors.get(impl, 'gray'),
                    marker=markers.get(impl, 'o'), markersize=4, linewidth=2)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Стоимость')
    ax.set_title('Функция стоимости')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # График 4: Время вычисления
    ax = axes[1, 1]
    box_data = []
    labels = []
    for impl, results in results_by_impl.items():
        if results:
            compute_times = []
            for result in results:
                if result.compute_times:
                    compute_times.extend(result.compute_times)
            box_data.append(compute_times)
            labels.append(impl)

    if box_data:
        ax.boxplot(box_data, labels=labels)
        ax.set_ylabel('Время вычисления (с)')
        ax.set_title('Распределение времени вычисления')
        ax.grid(True, alpha=0.3)

    # График 5: Успешность по реализациям
    ax = axes[2, 0]
    success_rates = []
    impl_labels = []
    for impl, results in results_by_impl.items():
        if results:
            success_count = sum(1 for r in results if r.success)
            success_rate = success_count / len(results)
            success_rates.append(success_rate)
            impl_labels.append(impl)

    if success_rates:
        bars = ax.bar(impl_labels, success_rates,
                      color=[colors.get(impl, 'gray') for impl in impl_labels])
        ax.set_ylabel('Успешность')
        ax.set_title('Успешность по реализациям')
        ax.set_ylim(0, 1.1)

        # Добавляем значения на столбцы
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom')

    # График 6: Количество шагов
    ax = axes[2, 1]
    num_steps_data = []
    for impl, results in results_by_impl.items():
        if results:
            steps = [len(r.time_steps) for r in results]
            num_steps_data.append(steps)

    if num_steps_data and labels:
        ax.boxplot(num_steps_data, labels=labels)
        ax.set_ylabel('Количество шагов')
        ax.set_title('Распределение количества шагов')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Графики сохранены в {save_path}")

    plt.show()


def generate_report(collector: ResultsCollector, output_file: str):
    """Генерирует HTML отчет по результатам"""
    stats = collector.get_statistics()

    html_template = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Отчет по результатам MPPI</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }}
            .stat-label {{
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                margin-top: 5px;
            }}
            .table-container {{
                overflow-x: auto;
                margin-bottom: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .success {{
                color: #28a745;
            }}
            .failed {{
                color: #dc3545;
            }}
            .timestamp {{
                font-size: 0.8em;
                color: #888;
                text-align: right;
                margin-top: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📊 Отчет по результатам экспериментов MPPI</h1>
            <p>Сгенерировано: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_results', 0)}</div>
                <div class="stat-label">Всего результатов</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_experiments', 0)}</div>
                <div class="stat-label">Экспериментов</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('success_stats', {{}}).get('success_rate', 0):.1%}</div>
                <div class="stat-label">Успешность</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(stats.get('implementations', {{}}))}</div>
                <div class="stat-label">Реализаций</div>
            </div>
        </div>

        <h2>Статистика по реализациям</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Реализация</th>
                        <th>Количество</th>
                        <th>Успешность</th>
                        <th>Среднее шагов</th>
                    </tr>
                </thead>
                <tbody>
    """

    for impl, impl_stats in stats.get('implementations', {}).items():
        html_template += f"""
                    <tr>
                        <td><strong>{impl.upper()}</strong></td>
                        <td>{impl_stats.get('count', 0)}</td>
                        <td class="success">{impl_stats.get('success_rate', 0):.1%}</td>
                        <td>{impl_stats.get('avg_steps', 0):.0f}</td>
                    </tr>
        """

    html_template += """
                </tbody>
            </table>
        </div>

        <h2>Статистика по экспериментам</h2>
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>Эксперимент</th>
                        <th>Количество</th>
                        <th>Успешность</th>
                    </tr>
                </thead>
                <tbody>
    """

    for exp, exp_stats in stats.get('experiments', {}).items():
        html_template += f"""
                    <tr>
                        <td>{exp}</td>
                        <td>{exp_stats.get('count', 0)}</td>
                        <td class="success">{exp_stats.get('success_rate', 0):.1%}</td>
                    </tr>
        """

    html_template += f"""
                </tbody>
            </table>
        </div>

        <div class="timestamp">
            Первый результат: {stats.get('timeline', {{}}).get('first_result', 'Н/Д')}<br>
            Последний результат: {stats.get('timeline', {{}}).get('last_result', 'Н/Д')}
        </div>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"Отчет сохранен в {output_file}")