import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np

# Классы:
- PendulumConfig
- InvertedPendulum

# Функции:
- derivatives(state, control, config)
- rk4_step(state, control, config, dt)