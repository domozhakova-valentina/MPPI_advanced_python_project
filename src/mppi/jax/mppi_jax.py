"""
Реализация MPPI на JAX с автоматическим дифференцированием
"""
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

import numpy as np
from ..base import MPPIBase


class MPPIJax(MPPIBase):
    """MPPI реализация на JAX"""
    
    def __init__(self, config: 'MPPIConfig'):
        if not JAX_AVAILABLE:
            raise ImportError("JAX не установлен. Установите: pip install jax jaxlib")
        
        super().__init__(config)
        self._compile_jax_functions()
    
    def _compile_jax_functions(self):
        """Компиляция JAX функций для ускорения"""
        # JAX версия динамики
        @jit
        def jax_dynamics(state, F):
            x, theta, dx, dtheta = state
            M = self.config.m_cart
            m = self.config.m_pole
            l = self.config.l
            g = self.config.g
            
            sin_theta = jnp.sin(theta)
            cos_theta = jnp.cos(theta)
            denom = M + m * sin_theta**2
            
            ddx = (F + m * sin_theta * (l * dtheta**2 + g * cos_theta)) / denom
            ddtheta = (-F * cos_theta - m * l * dtheta**2 * cos_theta * sin_theta - 
                      (M + m) * g * sin_theta) / (l * denom)
            
            return jnp.array([dx, dtheta, ddx, ddtheta])
        
        self._jax_dynamics = jax_dynamics
        
        # Функция стоимости в JAX
        @jit
        def jax_cost(state_traj, control_traj):
            # Векторизованное вычисление стоимости
            states = state_traj  # (T, 4)
            controls = control_traj  # (T,)
            
            # Штрафы за состояния
            state_cost = (self.config.Q[0] * states[:, 0]**2 +
                         self.config.Q[1] * states[:, 1]**2 +
                         self.config.Q[2] * states[:, 2]**2 +
                         self.config.Q[3] * states[:, 3]**2)
            
            # Штраф за управление
            control_cost = self.config.R * controls**2
            
            return jnp.sum(state_cost + control_cost)
        
        self._jax_cost = jax_cost
        
        # Векторизованная симуляция траектории
        @jit
        def simulate_trajectory(initial_state, controls):
            def step(carry, control):
                state = carry
                derivatives = jax_dynamics(state, control)
                next_state = state + derivatives * self.config.dt
                return next_state, next_state
            
            _, states = jax.lax.scan(step, initial_state, controls)
            return states
        
        self._simulate_trajectory = simulate_trajectory
    
    def compute_control(self, state: np.ndarray) -> float:
        """
        Вычисление управления с использованием JAX (векторизовано)
        """
        # Конвертация в JAX массивы
        state_jax = jnp.array(state)
        u_jax = jnp.array(self.u)
        
        # Генерация случайных возмущений
        key = jax.random.PRNGKey(0)
        epsilon = self.config.sigma * jax.random.normal(
            key, (self.config.K, self.config.T)
        )
        
        # Векторизованная симуляция всех траекторий
        def simulate_all(eps):
            controls = u_jax + eps
            states = vmap(self._simulate_trajectory, in_axes=(None, 0))(
                state_jax, controls
            )
            return states
        
        # Вычисление стоимостей для всех траекторий
        all_states = simulate_all(epsilon)
        all_controls = u_jax + epsilon
        
        costs = vmap(self._jax_cost)(all_states, all_controls)
        
        # Вычисление весов
        min_cost = jnp.min(costs)
        weights = jnp.exp(-(costs - min_cost) / self.config.lambda_)
        weights = weights / jnp.sum(weights)
        
        # Обновление траектории
        u_new = u_jax + jnp.sum(weights[:, jnp.newaxis] * epsilon, axis=0)
        self.u = np.array(u_new)
        
        # Сохранение истории
        self.costs_history.append(float(min_cost))
        
        return float(self.u[0])