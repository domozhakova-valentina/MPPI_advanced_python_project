"""
Тесты модуля визуализации
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend для тестов


class TestPlotter:
    """Тесты класса Plotter"""
    
    def test_plot_simulation_results(self, mock_history):
        """Тест построения графиков результатов"""
        from visualization.plots import Plotter
        import matplotlib.pyplot as plt
        
        # Создаем фигуру
        fig = Plotter.plot_simulation_results(mock_history)
        
        # Проверяем, что фигура создана
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 6  # 6 графиков в фигуре
        
        plt.close(fig)
        
    def test_plot_comparison(self):
        """Тест построения сравнения"""
        from visualization.plots import Plotter
        import matplotlib.pyplot as plt
        
        # Создаем тестовые данные
        metrics_dict = {
            'numpy': {'avg_angle': 0.1, 'max_angle': 0.5, 'execution_time': 1.0},
            'jax': {'avg_angle': 0.08, 'max_angle': 0.4, 'execution_time': 0.5},
            'cpp': {'avg_angle': 0.09, 'max_angle': 0.45, 'execution_time': 0.7}
        }
        
        # Создаем график
        fig = Plotter.plot_comparison(metrics_dict)
        
        # Проверяем, что фигура создана
        assert isinstance(fig, matplotlib.figure.Figure)
        
        plt.close(fig)
        
    def test_plot_performance_comparison(self):
        """Тест построения сравнения производительности"""
        from visualization.plots import Plotter
        import matplotlib.pyplot as plt
        
        # Создаем тестовые данные
        times_dict = {
            'numpy': 1.0,
            'jax': 0.5,
            'cpp': 0.7
        }
        
        # Создаем график
        fig = Plotter.plot_performance_comparison(times_dict)
        
        # Проверяем, что фигура создана
        assert isinstance(fig, matplotlib.figure.Figure)
        
        plt.close(fig)


class TestPendulumAnimator:
    """Тесты класса PendulumAnimator"""
    
    def test_animator_initialization(self, mock_history):
        """Тест инициализации аниматора"""
        from visualization.animate import PendulumAnimator
        
        animator = PendulumAnimator(
            history_states=mock_history['states'],
            history_controls=mock_history['controls'],
            pendulum_length=1.0
        )
        
        assert animator.history_states == mock_history['states']
        assert animator.history_controls == mock_history['controls']
        assert animator.pendulum_length == 1.0
        assert animator.fig is None
        assert animator.animation is None
        
    def test_create_animation(self, mock_history):
        """Тест создания анимации"""
        from visualization.animate import PendulumAnimator
        import matplotlib.animation as animation
        
        animator = PendulumAnimator(
            history_states=mock_history['states'],
            history_controls=mock_history['controls'],
            pendulum_length=1.0
        )
        
        # Создаем анимацию
        anim = animator.create_animation(interval=50)
        
        # Проверяем результат
        assert isinstance(anim, animation.FuncAnimation)
        assert animator.fig is not None
        assert animator.animation is not None
        
        # Проверяем, что фигура создана правильно
        assert len(animator.fig.axes) == 4  # 4 оси в фигуре
        
        # Очищаем
        import matplotlib.pyplot as plt
        plt.close(animator.fig)