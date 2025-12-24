class CombinedConfig:
    """Объединённая конфигурация для MPPI"""
    def __init__(self, pendulum: 'PendulumConfig', mppi: 'MPPIConfig'):
        self.m_cart = pendulum.m_cart
        self.m_pole = pendulum.m_pole
        self.l = pendulum.l
        self.g = pendulum.g
        self.dt = pendulum.dt
        
        self.K = mppi.K
        self.T = mppi.T
        self.lambda_ = mppi.lambda_
        self.sigma = mppi.sigma
        self.Q = mppi.Q
        self.R = mppi.R