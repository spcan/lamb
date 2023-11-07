# Experiment signal class.



# Import Numpy and Pandas.
import numpy as np
import pandas as pd

# Import internal data.
from .actuator import Actuator
from .sensor import Sensor

# Import signal processing.
from scipy.fft import fft, fftfreq
from scipy.signal import argrelextrema, hilbert, stft, get_window as window

# Import typing extensions.
from typing_extensions import Any, Self, Tuple


# Deactivate numpy deprecation warnings.
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 



class Signal:
    """Signal data obtained in a given sensor from a given actuator"""

    # End index of the source signal.
    endSource = 0

    # Start index of the sensor signal.
    startSignal = 0

    # True if the signal is already analyzed and cached.
    cached = False

    # The timestep of the signal (default 1 s).
    timestep = 1.0

    # The frequency of the signal data acquisition rate (default 1 Hz).
    frequency = 1.0


    def build(i: int, raw: Any, setup: Any) -> Self:
        """Builds the data class from the given experiment data"""

        # Create the class instance.
        self = Signal()

        # Store the index and get the data path setup.
        self.index = i

        # Store the actuator signal.
        self.actuator = Actuator.create( raw[0][0][0], raw[3][0], float( raw[4] ), float( raw[2] ) )

        # Store the sensor signal.
        self.sensor = Sensor( raw[1][0][0], raw[9][0][0] )

        # Store the signal frequency.
        self.frequency = setup[2].T[0][0]

        # Store the signal timestep in microseconds.
        self.timestep = 1.0 / (self.frequency / 1e6)

        # Store the signal sample average size.
        self.navg = setup[4].T[0][0]

        return self

    def compare(self, rhs: Self):
        """Compares this signal to another signal, creating a report"""

        # Check that both signals have the same actuator and sensor.
        if (self.sensor.id != rhs.sensor.id) or (self.actuator.id != rhs.actuator.id):
            return None

        # Ensure both signals have been analyzed.
        if not self.cached:
            self.analyze()

        if not rhs.cached:
            rhs.analyze()

        # Compare the signals.
        return Report().compare(self, rhs)

    def link(self, a, s):
        """Links a signal to its experiment data"""

        # Store the reference to the data.
        self.raw = a
        self.sen = s

        # Preprocess the actuator signal and zero it.
        smoothed = pd.Series( a.flatten() ).rolling(window=self.navg).mean()
        asymptote = np.average( smoothed[-100:-1] )
        self.act = a - asymptote

    def analyze(self, cache=True):
        """Analyzes the signal (timestamps, energy and frequency distributions, etc...)"""

        # Timestamp and analyze the signal spectrum.
        self.timestamp(cache)
        self.spectrum(cache)
        self.hilbert(cache)

        # Set the cached flag if enabled.
        if cache:
            self.cached = True

    def hilbert(self, cache=True):
        """Analyzes the hilbert envelope of the signal"""

        # Calculate the Hilbert signal.
        analytic = hilbert( self.getsensor().flatten() )
        envelope = np.abs( analytic )

        if cache:
            self.hil = envelope

    def spectrum(self, cache=True):
        """Analyzes the spectral energy distribution of the signal"""

        # Calculate the STFT.
        f, t, z = stft( self.getsensor().flatten(), window=window('hann', 4000 ), nperseg=4000, noverlap=3750, fs=self.frequency )
        #f, t, z = custom_stft( self.getsensor().flatten(), window=500, delta=100 )

        # Find all frequency indices below 1 MHz.
        idx = np.where( f < 6e5 )

        fftF = f[idx].copy()
        fftT = t
        fftZ = np.abs( z[idx] ).copy()

        # Get the amount of points in the signal.
        N = len(self.sen) - self.endSource
        S = self.endSource
        if (N % 2) != 0:
            N = N - 1
            S = S + 1

        # Calculate the full signal FFT.
        print(f"FFT on size {N} ({N/2})")
        fftFull  = fft( self.sen[S:] )
        fftFullX = fftfreq(N, 1 / self.frequency)[0:int(N/2)]
        fftFullY = (2.0 / N) * np.abs(fftFull[0:int(N/2)])

        # Select only frequencies under 1 MHz.
        fftFullY = fftFullY[ np.where(fftFullX < 1e6) ]
        fftFullX = fftFullX[ np.where(fftFullX < 1e6) ]

        # Get the FFT of the filtered frequencies.
        if cache:
            self.fftF = fftF
            self.fftT = fftT
            self.fftZ = fftZ

            self.fftFullX = fftFullX
            self.fftFullY = fftFullY

        return fftF, fftT, fftZ


    def timestamp(self, cache=True) -> Tuple[int, int]:
        """Timestamps the actuator and sensor signals of this experiment"""

        # Timestamp the source signal.
        endSource = self.timestampSource(cache)

        # Timestamp the sensor signal.
        startSignal = self.timestampSignal(endSource, cache)

        return endSource, startSignal

    def timestampSignal(self, endSource: int, cache: bool) -> int:
        """Calculates the timestamp of the start of the sensor signal"""

        # Find the maximum of the sensor signal to the right of the end of the source.
        absmax = np.argmax( self.sen[endSource:] )

        # Check if the array has enough size to split.
        if len( self.sen[endSource:endSource+absmax] ) < 500:
            startSignal = endSource
        else:
            # Calculate the STFT.
            #print(len(self.sen[endSource:endSource+absmax]))
            f, t, z = np.abs( stft(self.sen[endSource:endSource+absmax].flatten(), nperseg=150, fs=self.frequency) )

            # Find the index of the maximum value of Z.
            [ixmax, iymax] = np.unravel_index( np.argmax( z ), z.shape )

            # Get the FFT values on the frequency with the maximum, divide by 10 and cast to integer to truncate.
            fftmax = np.divide( z[ixmax, :], 10.0 ).astype( int )

            # Get the places where the fftmax is 0.
            zeroes = np.where( fftmax[:iymax] == 0 )[0]

            # Check if there are zeroes in the scaled FFT values.
            if len(zeroes) > 0:
                abslow = zeroes[-1]
            else:
                abslow = np.argmin( fftmax )

            # Get the index of the start of signal.
            startSignal = int(t[abslow] * self.frequency) + endSource

        if cache:
            # Store the start signal index.
            self.startSignal = startSignal

            # Store the index of the maximum.
            self.sigmax = absmax + endSource

    def timestampSource(self, cache: bool) -> int:
        """Calculates the timestamp of the end of the source signal"""

        # Create the envelope of the actuator signal.
        _, imax = envelopes( pd.Series( np.abs( self.act ).flatten() ).rolling(20).mean().to_numpy(copy=False) )

        # Get the maximums of the signal.
        lmax = np.abs( self.act[imax] )

        # Get the absolute maximum.
        absmax = np.argmax( lmax )

        # Get the first minimum after the maximum.
        absmin = argrelextrema( lmax[absmax+1:], np.less )[0][0] + 1

        # Get the first element under 20 mV.
        abslow = np.where( pd.Series( lmax[absmax+1:].flatten() ).rolling(5).mean().to_numpy() < 15 )[0][0] + 1
        endSourcea = imax[absmax + abslow]

        # Get the first minimum element after the maximum.
        endSourceb = imax[absmax + absmin]

        # Store the calculated values.
        if cache:
            # Select as end source index the best fit.
            # Currently this fit is A.
            self.endSource = endSourcea

            # Store the index of the end of signal.
            self.endSourcea = endSourcea
            self.endSourceb = endSourceb

        return endSourcea

    def timeaxis(self):
        """Returns the time axis of this signal"""

        return np.linspace(0, len(self.s)) * self.timestep

    def getsensor(self, filtered: bool = True):
        """Returns the (optionally) filtered sensor signal"""

        # Early return if no filter is needed.
        if not filtered:
            return self.sen

        # Copy the array.
        fil = self.sen

        # Zero out the source signal.
        fil[:self.endSource] = 0

        return fil

    def report(self):
        """Prints to STDOUT a report of the signal"""

        # Report the signal ID.
        print(f"Signal {self.index}")
        print(f"  {len(self.act)} data points")
        print(f"  Source:")
        print(F"    ID   {self.actuator.id}")
        print(F"    Type {self.actuator.signal}")
        print(F"    Freq {self.actuator.freq / 1000.0} kHz")
        print(F"    Volt {self.actuator.volts} V")

        print(f"  Input:")
        print(F"    ID   {self.sensor.id}")
        print(F"    Gain {self.sensor.gain} dB")




class Report:
    """Contains the comparison report between two signals."""

    # Actuator ID.
    act = 0

    # Sensor ID.
    sen = 0

    # The delay between signal start (in microseconds).
    delay = 0

    # The frequency of the source signal.
    srcfreq = 0.0

    def compare(self, lhs: Signal, rhs: Signal) -> Self:
        """Compares two signals and generates a report"""

        # Store the actuator and sensor IDs.
        self.act = lhs.actuator.id
        self.sen = lhs.sensor.id

        # Store the start time of both signals.
        self.ssr = rhs.startSignal
        self.ssl = lhs.startSignal

        # Store the timestep of both signals.
        self.tsr = rhs.timestep
        self.tsl = lhs.timestep

        # Save the source frequency.
        self.sfr = rhs.actuator.freq
        self.sfl = lhs.actuator.freq

        # Compare the travel time of the actuator signal.
        self.delay = (rhs.startSignal * rhs.timestep) - (lhs.startSignal * lhs.timestep)

        return self



def envelopes(s, dmin=1, dmax=1, split=False):
    """
    Input :
    s: 1d-array, data signal from which to extract high and low envelopes
    dmin, dmax: int, optional, size of chunks, use this if the size of the input signal is too big
    split: bool, optional, if True, split the signal in half along its mean, might help to generate the envelope in some cases
    Output :
    lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    
    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]

    # global min of dmin-chunks of locals min 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global max of dmax-chunks of locals max 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return lmin,lmax


def custom_stft (y , window =2000 , delta =200):

    """ Function to create a spectrogram based in the STFT algorithm
    but simplyfied

    : param y (1 D np . array or list ) : time series signal

    : optional window ( int ) : number of points taken to compute the FFT
    : optional delta ( int ) : hop length between windows

    : return stft (2 D np . array ): spectrogram

    """

    window = int( window / 2 )

    steps = range( window , len( y ) - window +1 , delta )
    stft = list()
    for i in steps :

        yy = y[i - window : i + window ]

        YY = np.fft.fft( yy )
        stft.append( YY )
        stft = np.array( stft )

    return stft.T
