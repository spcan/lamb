# Sensor signal class.



from typing_extensions import Self



class Actuator:
    """Common class abstraction for the source signal"""

    def create(id: int, signal: str, frequency: float, voltage: float) -> Self:
        """Creates a new instance of the class"""

        # Create the instance.
        self = Actuator()

        # Store the ID, signal type and frequency.
        self.id     = id
        self.signal = signal
        self.freq   = frequency
        self.volts  = voltage

        return self
