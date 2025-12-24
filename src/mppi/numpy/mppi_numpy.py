"""
Реализация MPPI на чистом NumPy
"""
import numpy as np
from ..base import MPPIBase
from typing import List


class MPPINumpy(MPPIBase):
    """MPPI реализация на NumPy"""
    
    def compute_control(self, state: np.ndarray) -> float:
        """
        Вычисление управления с использованием NumPy
        """
        # Генерация случайных возмущений
        epsilon = self.config.sigma * np.random.randn(self.config.K, self.config.T)
        
        # Копирование текущей траектории для всех сэмплов
        u_expanded = self.u + np.zeros((self.config.K, self.config.T))
        
        # Стоимости для каждого сэмпла
        costs = np.zeros(self.config.K)
        
        # Оценка стоимости для каждой траектории
        for k in range(self.config.K):
            # Пробная траектория управления
            u_sample = u_expanded[k] + epsilon[k]
            
            # Симуляция траектории
            state_traj = np.zeros((self.config.T, 4))
            current_state = state.copy()
            
            for t in range(self.config.T):
                state_traj[t] = current_state
                # Интегрирование динамики (метод Эйлера)
                derivatives = self._dynamics(current_state, u_sample[t])
                current_state = current_state + derivatives * self.config.dt
            
            # Вычисление стоимости
            costs[k] = self._cost_function(state_traj, u_sample)
        
        # Вычисление весов
        min_cost = np.min(costs)
        weights = np.exp(-(costs - min_cost) / self.config.lambda_)
        weights = weights / np.sum(weights)  # нормализация
        
        # Обновление оптимальной траектории
        self.u = self.u + np.sum(weights[:, np.newaxis] * epsilon, axis=0)
        
        # Сохранение истории стоимостей
        self.costs_history.append(min_cost)
        
        # Возврат первого управления
        return float(self.u[0])