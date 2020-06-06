import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy import signal
import sys
from typing import Text, Tuple

from typing import Sequence


def read_wav_file(file: Text) -> Tuple[int, np.ndarray]:
    """
    Reads a wave file and returns it as a NumPy array.

    Args:
       file:  Filepath to a .wav file.

    Returns: (samprate, data) where samprate is the sampling frequency and data
    is a numpy array with dtype int16 and shape (num_channels, num_samples).

    Raises:
      RuntimeError: if an error occurred while reading the data.
      wave.Error: whatever errors the wave module encountered
      OsError (via wave module), if a file could not be opened.
    """

    wave_reader = wave.Wave_read(file)
    (nchannels, sampwidth, framerate,
     nframes, comptype, compname) = wave_reader.getparams()
    if comptype != 'NONE':
        raise RuntimeError(
            "Wave file has compression, which is unsupported: comptype={},"
            "compname={}".format(comptype, compname))
    # Expect 16-bit  magnitude sampling.
    if sampwidth != 2:
        raise RuntimeError(
            "Wave file has sample width of {}, expected 2.".format(
                sampwidth))
    data_as_bytes = wave_reader.readframes(nframes)
    nframes_read = len(data_as_bytes) // (sampwidth * nchannels)
    assert nframes_read <= nframes
    dt = np.dtype('int16')
    if sys.byteorder == 'big':
        # Make sure to interpret the data as little-endian even if the machine
        # is big endian.
        dt = dt.newbyteorder('<')
    array = np.frombuffer(data_as_bytes, dt)
    # order='F' because the frame has a higher stride than the channel.
    return framerate, array.reshape((nchannels, nframes_read), order='F')


def force_wav_data_to_mono(wav_data: np.ndarray):
    """ Converts two channel wav data to a single channel"""
    ndims = len(wav_data.shape)
    if ndims == 1:
        return wav_data
    elif ndims == 2:
        return wav_data[0, :]
    else:
        raise ValueError("Wav file  is neither mono nor stereo")


class Sound:
    def __init__(self, time_series: Sequence, sampling_freq: int):
        self._time_series = np.asarray(time_series)
        self._sampling_freq = sampling_freq

    @classmethod
    def sound_from_wav(cls, wav_path):
        sampling_freq, wav_array = read_wav_file(wav_path)
        wav_array = force_wav_data_to_mono(wav_array)
        return Sound(wav_array, sampling_freq)

    # TODO(kane): Add support for spectrogram configuration
    def compute_spectrogram(self):
        return signal.spectrogram(self._time_series, self._sampling_freq)

    # TODO(kane): Add support for plot configuration
    def plot_spectrogram(self):
        f, t, Sxx = self.compute_spectrogram()
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.ylim(0, 1000)
        plt.show()
