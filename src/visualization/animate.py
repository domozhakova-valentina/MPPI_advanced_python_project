"""
Модуль анимации для визуализации работы маятника.

Паттерны:
- Observer: обновление анимации в реальном времени
- Builder: построение сложных анимаций
- Strategy: разные стили анимации
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import warnings
from IPython.display import HTML, display
import ipywidgets as widgets
from pathlib import Path
import time
from enum import Enum

from ..controller.config import State, SystemConfig


class AnimationStyle(Enum):
    """Стили анимации"""
    SIMPLE = "simple"  # Простая схематичная анимация
    REALISTIC = "realistic"  # Реалистичная анимация с текстурами
    SCIENTIFIC = "scientific"  # Научный стиль с осями и метками
    MINIMAL = "minimal"  # Минималистичный стиль


@dataclass
class AnimationConfig:
    """Конфигурация анимации"""
    style: AnimationStyle = AnimationStyle.SIMPLE
    fps: int = 30
    interval: int = 50  # Интервал между кадрами в миллисекундах
    save_path: Optional[str] = None
    show_grid: bool = True
    show_controls: bool = True
    show_trajectory: bool = True
    track_length: float = 5.0  # Длина трека для тележки
    scale: float = 1.0  # Масштаб анимации
    background_color: str = "#f5f5f5"
    cart_color: str = "#2c3e50"
    pole_color: str = "#e74c3c"
    trajectory_color: str = "#3498db"
    control_color: str = "#2ecc71"

    def __post_init__(self):
        """Валидация конфигурации"""
        if self.fps <= 0:
            raise ValueError("FPS должен быть положительным")
        if self.interval <= 0:
            raise ValueError("Интервал должен быть положительным")
        if self.track_length <= 0:
            raise ValueError("Длина трека должна быть положительной")
        if self.scale <= 0:
            raise ValueError("Масштаб должен быть положительным")


class PendulumAnimator:
    """Класс для создания анимаций маятника

    Паттерн: Builder - поэтапное построение анимации
    """

    def __init__(self, config: Optional[AnimationConfig] = None):
        """Инициализирует аниматор

        Args:
            config: конфигурация анимации
        """
        self.config = config or AnimationConfig()
        self.fig = None
        self.ax = None
        self.anim = None

        # Элементы анимации
        self.cart_patch = None
        self.pole_line = None
        self.pendulum_circle = None
        self.trajectory_line = None
        self.control_bar = None
        self.control_text = None
        self.angle_text = None
        self.time_text = None

        # Данные
        self.trajectory: List[State] = []
        self.controls: List[float] = []
        self.time_steps: List[float] = []

    def set_data(self, trajectory: List[State],
                 controls: Optional[List[float]] = None,
                 time_steps: Optional[List[float]] = None):
        """Устанавливает данные для анимации

        Args:
            trajectory: список состояний
            controls: список управлений (опционально)
            time_steps: список временных шагов (опционально)
        """
        self.trajectory = trajectory

        if controls is not None:
            self.controls = controls
        else:
            self.controls = [0.0] * len(trajectory)

        if time_steps is not None:
            self.time_steps = time_steps
        else:
            self.time_steps = list(range(len(trajectory)))

        return self

    def _create_figure(self):
        """Создает фигуру для анимации"""
        fig, ax = plt.subplots(figsize=(12, 8))

        # Настраиваем оси
        track_length = self.config.track_length
        scale = self.config.scale

        ax.set_xlim(-track_length / 2, track_length / 2)
        ax.set_ylim(-track_length * 0.4, track_length * 0.6)
        ax.set_aspect('equal')
        ax.set_facecolor(self.config.background_color)

        if self.config.show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)

        # Рисуем трек
        track_height = -0.1
        ax.add_patch(Rectangle(
            (-track_length / 2, track_height - 0.05),
            track_length, 0.1,
            facecolor='#7f8c8d', edgecolor='#2c3e50', linewidth=2
        ))

        # Отметки на треке
        for x in np.linspace(-track_length / 2, track_length / 2, 11):
            ax.plot([x, x], [track_height - 0.05, track_height - 0.1],
                    'k-', linewidth=1)
            if abs(x) < track_length / 2 - 0.5:
                ax.text(x, track_height - 0.15, f'{x:.1f}',
                        ha='center', va='top', fontsize=8)

        self.fig = fig
        self.ax = ax

        return self

    def _create_elements(self):
        """Создает графические элементы анимации"""
        # Тележка (прямоугольник)
        cart_width = 0.4 * self.config.scale
        cart_height = 0.2 * self.config.scale
        self.cart_patch = Rectangle(
            (-cart_width / 2, -cart_height / 2),
            cart_width, cart_height,
            facecolor=self.config.cart_color,
            edgecolor='black',
            linewidth=2,
            zorder=10
        )
        self.ax.add_patch(self.cart_patch)

        # Маятник (линия и круг на конце)
        self.pole_line, = self.ax.plot([], [],
                                       color=self.config.pole_color,
                                       linewidth=3,
                                       zorder=9)

        self.pendulum_circle = Circle(
            (0, 0), 0.05 * self.config.scale,
            facecolor=self.config.pole_color,
            edgecolor='black',
            linewidth=2,
            zorder=11
        )
        self.ax.add_patch(self.pendulum_circle)

        # Траектория (если включена)
        if self.config.show_trajectory and len(self.trajectory) > 0:
            self.trajectory_line, = self.ax.plot([], [],
                                                 color=self.config.trajectory_color,
                                                 linewidth=1.5,
                                                 linestyle='--',
                                                 alpha=0.7,
                                                 zorder=5)

        # График управления (если включен)
        if self.config.show_controls:
            # Создаем вторую ось для управления
            self.control_ax = self.ax.inset_axes([0.02, 0.02, 0.3, 0.2])
            self.control_ax.set_facecolor('white')
            self.control_ax.set_title('Управление (Н)', fontsize=10)
            self.control_ax.set_xlabel('Время (с)', fontsize=8)
            self.control_ax.set_ylabel('Сила', fontsize=8)
            self.control_ax.grid(True, alpha=0.3)

            self.control_line, = self.control_ax.plot([], [],
                                                      color=self.config.control_color,
                                                      linewidth=2)
            self.control_point, = self.control_ax.plot([], [], 'o',
                                                       color=self.config.control_color,
                                                       markersize=8)

            # Ограничиваем диапазон оси Y
            if self.controls:
                max_control = max(abs(min(self.controls)), abs(max(self.controls)))
                self.control_ax.set_ylim(-max_control * 1.2, max_control * 1.2)

        # Текстовые элементы
        self.time_text = self.ax.text(
            0.02, 0.98, 'Время: 0.00 с',
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.angle_text = self.ax.text(
            0.98, 0.98, 'Угол: 0.0°',
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        self.control_text = self.ax.text(
            0.02, 0.92, 'Управление: 0.00 Н',
            transform=self.ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        return self

    def _update_animation(self, frame: int):
        """Обновляет анимацию для заданного кадра

        Args:
            frame: номер кадра

        Returns:
            список обновленных элементов
        """
        if frame >= len(self.trajectory):
            return []

        state = self.trajectory[frame]
        control = self.controls[frame] if frame < len(self.controls) else 0.0
        time_val = self.time_steps[frame] if frame < len(self.time_steps) else frame

        # Параметры системы
        cart_width = 0.4 * self.config.scale
        pole_length = 1.0 * self.config.scale

        # Положение тележки
        cart_x = state.x
        cart_y = 0.0

        # Обновляем тележку
        self.cart_patch.set_x(cart_x - cart_width / 2)

        # Положение маятника
        pendulum_x = cart_x + pole_length * np.sin(state.theta)
        pendulum_y = cart_y + pole_length * np.cos(state.theta)

        # Обновляем маятник
        self.pole_line.set_data([cart_x, pendulum_x], [cart_y, pendulum_y])
        self.pendulum_circle.center = (pendulum_x, pendulum_y)

        # Обновляем траекторию
        if self.config.show_trajectory and self.trajectory_line is not None:
            if frame > 0:
                # Добавляем точку к траектории
                xs = [state.x for state in self.trajectory[:frame + 1]]
                ys = [state.x_dot for state in self.trajectory[:frame + 1]]  # Можно менять на нужную переменную
                self.trajectory_line.set_data(xs, ys)

        # Обновляем график управления
        if self.config.show_controls:
            if frame > 0:
                times = self.time_steps[:frame + 1]
                controls = self.controls[:frame + 1]
                self.control_line.set_data(times, controls)
                self.control_point.set_data([time_val], [control])

        # Обновляем текстовые элементы
        self.time_text.set_text(f'Время: {time_val:.2f} с')
        self.angle_text.set_text(f'Угол: {np.degrees(state.theta):.1f}°')
        self.control_text.set_text(f'Управление: {control:.2f} Н')

        # Собираем все элементы для обновления
        elements = [self.cart_patch, self.pole_line, self.pendulum_circle,
                    self.time_text, self.angle_text, self.control_text]

        if self.config.show_trajectory and self.trajectory_line is not None:
            elements.append(self.trajectory_line)

        if self.config.show_controls:
            elements.extend([self.control_line, self.control_point])

        return elements

    def create(self) -> animation.FuncAnimation:
        """Создает анимацию

        Returns:
            объект анимации matplotlib
        """
        if not self.trajectory:
            raise ValueError("Нет данных для анимации. Используйте set_data()")

        # Создаем фигуру и элементы
        self._create_figure()
        self._create_elements()

        # Создаем анимацию
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            frames=len(self.trajectory),
            interval=self.config.interval,
            blit=True,
            repeat=False
        )

        return self.anim

    def show(self):
        """Показывает анимацию"""
        if self.anim is None:
            self.create()

        plt.close(self.fig)  # Закрываем старую фигуру
        return HTML(self.anim.to_jshtml())

    def save(self, filename: str, fps: Optional[int] = None):
        """Сохраняет анимацию в файл

        Args:
            filename: имя файла для сохранения
            fps: частота кадров (если None, используется из конфигурации)
        """
        if self.anim is None:
            self.create()

        save_fps = fps or self.config.fps

        # Поддерживаемые форматы
        supported_formats = {
            '.gif': 'pillow',
            '.mp4': 'ffmpeg',
            '.avi': 'ffmpeg',
            '.mov': 'ffmpeg'
        }

        ext = Path(filename).suffix.lower()
        if ext not in supported_formats:
            raise ValueError(f"Неподдерживаемый формат: {ext}. "
                             f"Поддерживаемые: {list(supported_formats.keys())}")

        writer = supported_formats[ext]
        self.anim.save(filename, writer=writer, fps=save_fps)
        print(f"Анимация сохранена в {filename}")

    def close(self):
        """Закрывает анимацию и освобождает ресурсы"""
        if self.anim:
            self.anim.event_source.stop()
        if self.fig:
            plt.close(self.fig)
        self.anim = None
        self.fig = None
        self.ax = None


def create_animation(trajectory: List[State],
                     controls: Optional[List[float]] = None,
                     config: Optional[AnimationConfig] = None,
                     **kwargs) -> PendulumAnimator:
    """Создает анимацию маятника

    Args:
        trajectory: список состояний
        controls: список управлений
        config: конфигурация анимации
        **kwargs: дополнительные параметры для конфигурации

    Returns:
        аниматор
    """
    if config is None:
        config = AnimationConfig(**kwargs)

    animator = PendulumAnimator(config)
    animator.set_data(trajectory, controls)

    return animator


def save_animation(animator: PendulumAnimator, filename: str, fps: Optional[int] = None):
    """Сохраняет анимацию в файл

    Args:
        animator: аниматор
        filename: имя файла
        fps: частота кадров
    """
    animator.save(filename, fps)


def real_time_animation(controller,
                        duration: float = 10.0,
                        update_interval: float = 0.1,
                        config: Optional[AnimationConfig] = None):
    """Создает анимацию в реальном времени

    Args:
        controller: контроллер MPPI
        duration: продолжительность анимации (с)
        update_interval: интервал обновления (с)
        config: конфигурация анимации

    Returns:
        аниматор
    """
    # Создаем аниматор
    if config is None:
        config = AnimationConfig(style=AnimationStyle.REALISTIC)

    animator = PendulumAnimator(config)

    # Создаем фигуру заранее
    animator._create_figure()
    animator._create_elements()

    # Подготавливаем данные
    trajectory = []
    controls = []
    time_steps = []

    # Функция обновления в реальном времени
    def update_real_time(frame):
        nonlocal trajectory, controls, time_steps

        # Вычисляем новое состояние
        current_state = trajectory[-1] if trajectory else State()
        control = controller.compute_control(current_state)
        next_state = controller.model.step(current_state, control,
                                           controller.system_config.dt)

        # Сохраняем данные
        trajectory.append(next_state)
        controls.append(control)
        time_steps.append(frame * update_interval)

        # Обновляем анимацию
        animator.set_data(trajectory, controls, time_steps)
        return animator._update_animation(frame)

    # Создаем анимацию
    num_frames = int(duration / update_interval)
    anim = animation.FuncAnimation(
        animator.fig,
        update_real_time,
        frames=num_frames,
        interval=update_interval * 1000,  # в миллисекундах
        blit=True,
        repeat=False
    )

    animator.anim = anim
    return animator


def create_interactive_animation():
    """Создает интерактивную анимацию с виджетами управления"""
    # Создаем виджеты
    style_dropdown = widgets.Dropdown(
        options=[(style.value, style) for style in AnimationStyle],
        value=AnimationStyle.SIMPLE,
        description='Стиль:'
    )

    fps_slider = widgets.IntSlider(
        value=30,
        min=1,
        max=60,
        step=1,
        description='FPS:'
    )

    scale_slider = widgets.FloatSlider(
        value=1.0,
        min=0.5,
        max=2.0,
        step=0.1,
        description='Масштаб:'
    )

    show_trajectory = widgets.Checkbox(
        value=True,
        description='Показывать траекторию'
    )

    show_controls = widgets.Checkbox(
        value=True,
        description='Показывать управление'
    )

    # Создаем выход для анимации
    output = widgets.Output()

    # Функция обновления
    def update_animation(style, fps, scale, show_traj, show_ctrl):
        with output:
            output.clear_output()

            # Создаем тестовые данные
            time = np.linspace(0, 10, 200)
            trajectory = [
                State(
                    x=0.5 * np.sin(t),
                    theta=0.3 * np.sin(2 * t),
                    x_dot=0.5 * np.cos(t),
                    theta_dot=0.6 * np.cos(2 * t)
                )
                for t in time
            ]

            controls = [2.0 * np.sin(t) for t in time]

            # Создаем анимацию
            config = AnimationConfig(
                style=style,
                fps=fps,
                scale=scale,
                show_trajectory=show_traj,
                show_controls=show_ctrl
            )

            animator = create_animation(trajectory, controls, config)
            display(animator.show())

    # Связываем виджеты
    widgets.interactive(
        update_animation,
        style=style_dropdown,
        fps=fps_slider,
        scale=scale_slider,
        show_traj=show_trajectory,
        show_ctrl=show_controls
    )

    # Показываем виджеты
    controls_box = widgets.VBox([
        style_dropdown,
        fps_slider,
        scale_slider,
        show_trajectory,
        show_controls
    ])

    return widgets.VBox([controls_box, output])


# Пример использования
if __name__ == "__main__":
    print("Testing Animation Module")
    print("=" * 60)

    # Создаем тестовые данные
    num_points = 100
    time = np.linspace(0, 10, num_points)

    trajectory = []
    controls = []

    for t in time:
        # Простая синусоидальная траектория
        state = State(
            x=0.5 * np.sin(t),
            theta=0.3 * np.sin(2 * t),
            x_dot=0.5 * np.cos(t),
            theta_dot=0.6 * np.cos(2 * t)
        )
        trajectory.append(state)
        controls.append(2.0 * np.sin(t))

    print(f"Создано {len(trajectory)} точек траектории")

    # Тестируем разные стили
    for style in AnimationStyle:
        print(f"\nТестируем стиль: {style.value}")

        try:
            config = AnimationConfig(
                style=style,
                fps=15,  # Низкий FPS для тестирования
                interval=100,
                show_grid=True,
                show_trajectory=True,
                show_controls=True
            )

            animator = create_animation(trajectory, controls, config)
            print(f"  ✓ Анимация создана успешно")

            # Сохраняем в файл (только для некоторых форматов)
            if style == AnimationStyle.SIMPLE:
                animator.save(f"test_animation_{style.value}.gif", fps=10)
                print(f"  ✓ Анимация сохранена как test_animation_{style.value}.gif")

            animator.close()
            print(f"  ✓ Ресурсы освобождены")

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

    print("\n" + "=" * 60)
    print("Animation module tested successfully!")
    print("=" * 60)