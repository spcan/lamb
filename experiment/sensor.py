# Sensor signal class.



from dataclasses import dataclass
from typing_extensions import Self



@dataclass
class Sensor:
    """Common class abstraction for the sensor signal"""
    id: int
    gain: float
