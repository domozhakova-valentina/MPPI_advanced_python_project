"""
Анимация маятника
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from typing import List, Optional


class PendulumAnimator:
    """Класс для создания анимации маятника"""
    
    def __init__(self, 
                 history_states: List[np.ndarray],
                 history_controls: List[float],
                 pendulum_length: float = 1.0):
        """
        Args:
            history_states: список состояний системы
            history_controls: список приложенных сил
            pendulum_length: длина маятника
        """
        self.history_states = history_states
        self.history_controls = history_controls
        self.pendulum_length = pendulum_length
        self.fig = None
        self.animation = None
    
    def create_animation(self, interval: int = 50) -> animation.FuncAnimation:
        """
        Создание анимации
        
        Args:
            interval: интервал между кадрами в мс
            
        Returns:
            Объект анимации
        """
        self.fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Настройка осей
        ax1.set_xlim(-2.5, 2.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect('equal')
        ax1.set_title('Маятник')
        ax1.grid(True)
        
        # Инициализация графических элементов
        cart_width = 0.4
        cart_height = 0.2
        
        # Тележка
        cart = plt.Rectangle((-cart_width/2, -cart_height/2), 
                            cart_width, cart_height, 
                            fc='blue', alpha=0.7)
        ax1.add_patch(cart)
        
        # Маятник
        pendulum_line, = ax1.plot([], [], 'r-', lw=3)
        pendulum_mass, = ax1.plot([], [], 'ro', markersize=10)
        
        # Графики
        time_ax = list(range(len(self.history_states)))
        
        # Угол
        ax2.set_xlim(0, len(self.history_states))
        ax2.set_ylim(-np.pi/2, np.pi/2)
        ax2.set_xlabel('Шаг')
        ax2.set_ylabel('Угол (рад)')
        ax2.grid(True)
        angle_line, = ax2.plot([], [], 'b-', lw=2)
        
        # Сила
        ax3.set_xlim(0, len(self.history_controls))
        ax3.set_ylim(min(self.history_controls) - 1, 
                    max(self.history_controls) + 1)
        ax3.set_xlabel('Шаг')
        ax3.set_ylabel('Сила (Н)')
        ax3.grid(True)
        force_line, = ax3.plot([], [], 'g-', lw=2)
        
        # Положение тележки
        ax4.set_xlim(0, len(self.history_states))
        ax4.set_xlabel('Шаг')
        ax4.set_ylabel('Положение (м)')
        ax4.grid(True)
        position_line, = ax4.plot([], [], 'm-', lw=2)
        
        def init():
            cart.set_xy((-cart_width/2, -cart_height/2))
            pendulum_line.set_data([], [])
            pendulum_mass.set_data([], [])
            angle_line.set_data([], [])
            force_line.set_data([], [])
            position_line.set_data([], [])
            return cart, pendulum_line, pendulum_mass, angle_line, force_line, position_line
        
        def update(frame):
            state = self.history_states[frame]
            x, theta = state[0], state[1]
            
            # Обновление тележки
            cart.set_x(x - cart_width/2)
            
            # Обновление маятника
            pendulum_x = x + self.pendulum_length * np.sin(theta)
            pendulum_y = self.pendulum_length * np.cos(theta)
            pendulum_line.set_data([x, pendulum_x], [0, pendulum_y])
            pendulum_mass.set_data([pendulum_x], [pendulum_y])
            
            # Обновление графиков
            angle_line.set_data(time_ax[:frame+1], 
                               [s[1] for s in self.history_states[:frame+1]])
            force_line.set_data(time_ax[:frame+1], 
                               self.history_controls[:frame+1])
            position_line.set_data(time_ax[:frame+1], 
                                  [s[0] for s in self.history_states[:frame+1]])
            
            return cart, pendulum_line, pendulum_mass, angle_line, force_line, position_line
        
        self.animation = animation.FuncAnimation(
            self.fig, update, frames=len(self.history_states),
            init_func=init, blit=True, interval=interval
        )
        
        plt.tight_layout()
        return self.animation
    
    def save(self, filename: str, fps: int = 30):
        """Сохранение анимации в файл"""
        if self.animation is None:
            raise ValueError("Сначала создайте анимацию")
        
        self.animation.save(filename, writer='ffmpeg', fps=fps)
    
    def display(self):
        """Отображение анимации в Jupyter"""
        from IPython.display import HTML
        return HTML(self.animation.to_jshtml())
