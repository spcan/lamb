# Wrapper of the Accelent Matlab data.



import scipy.io as sio

from .expsignal import Signal
from dataclasses import dataclass
from multiprocessing import Pool
from typing_extensions import Self



class SingleExperimentData:
    """Data obtained at a single frequency during and experiment"""

    # List of all signals in this experiment.
    signals = []

    # name of this experiment.
    name = ""

    def parse(raw) -> Self:
        """Parses the given Matlab data for Accelent data"""

        # Create the output instance.
        self = SingleExperimentData()

        # Store the setup information.
        self.setup = raw['setup'][0][0]

        # Create the list of signals in this experiment.
        self.signals = list()

        # Parse all signal headers.
        for i, header in enumerate( self.setup[6][0] ):
            self.signals.append( Signal.build(i, header, self.setup) )

        # Link all signal headers to their data.
        for i, signal in enumerate( self.signals ):
            signal.link( raw[f'a{i}'], raw[f's{i}'] )

        return self

    def load(file: str) -> Self:
        """Loads and parses the given file for Accelent data"""

        # Load the data through Scipy.
        raw = sio.loadmat( file )

        return SingleExperimentData.parse( raw )

    def analyze(self):
        """Analyzes all the signals in the experiment"""
        for signal in self.signals:
            signal.analyze()

    def compare(self, rhs: Self):
        """Compares this experiment's data with another experiment's data"""

        # Create a process pool and a list of all results.
        results = list()

        # Compare all signals to one another.
        for s in self.signals:
            for o in rhs.signals:
                # Compare the signals.
                result = s.compare(o)

                # Check if the copmarison was successful.
                if result is not None:
                    results.append(result)

        return Report().aggregate( results )

    def samples(self):
        """Returns the number of samples per array of this experiment"""
        return self.setup[3].T[0][0]

    def frequency(self) -> int:
        """Returns the sample rate of this experiment"""
        return self.setup[2].T[0][0]

    def navg(self):
        """Returns the number of samples averaged for each data point"""
        return self.setup[4].T[0][0]

    def timestep(self):
        """Returns the timestep between each data point in microseconds. NOTE : The instrument uses a rolling average"""
        return 1.0 / (self.frequency() / 1e6)

    def rename(self, name: str):
        self.name = name




class Report:
    """Report of a comparison between two experiments"""

    # List of all reports that could be generated.
    reports = []

    def aggregate(self, reports: list) -> Self:
        """Associates this report with all the given individual signal reports"""

        self.reports = reports

        return self

    def pairs(self) -> list:
        """Returns a list which contains all the reports between an actuator-sensor pair"""

        map = {}

        for report in self.reports:
            # Create the key.
            key = (report.act, report.sen)

            # Initialize if it doesn't exist.
            if not key in map:
                map[key] = []

            map[key].append( report )

        return [l for l in map.values()]
