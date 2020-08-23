from google.protobuf import text_format
import matplotlib.pyplot as plt
import numpy as np
import wave

from scipy import signal as sig
from masking_analysis.protos import sound_pb2, sound_generation_pb2
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


def gen_pure_tone_time_series(sound_gen_config):
  x = np.arange(
    sound_gen_config.duration * sound_gen_config.fs) / sound_gen_config.fs
  return np.sin(2 * np.pi * sound_gen_config.pure_tone_config.center_freq * x)


def _filter_band(time_series, fs, start_freq, stop_freq, filter_order):
  nyquist_freq = fs / 2
  start_freq_ratio = start_freq / nyquist_freq
  stop_freq_ratio = stop_freq / nyquist_freq
  sos = sig.butter(filter_order, [start_freq_ratio, stop_freq_ratio],
                   btype='bandpass', output='sos')
  return sig.sosfilt(sos, time_series)


def gen_flat_spectrum_time_series(sound_gen_config):
  white_noise = [np.random.normal() for _ in
                 range(sound_gen_config.duration * sound_gen_config.fs)]
  return _filter_band(white_noise, sound_gen_config.fs,
                      sound_gen_config.flat_spectrum_noise_config.start_freq,
                      sound_gen_config.flat_spectrum_noise_config.stop_freq,
                      sound_gen_config.flat_spectrum_noise_config.filter_order)


class Sound:
  _time_series: np.ndarray

  def __init__(self, time_series: Sequence, sampling_freq: int):
    self._time_series = np.asarray(time_series)
    self._times = [t / sampling_freq for t in range(0, len(time_series))]
    self._sampling_freq = sampling_freq

  @property
  def time_series(self):
    return self._time_series

  @classmethod
  def sound_from_wav(cls, wav_path):
    sampling_freq, wav_array = read_wav_file(wav_path)
    wav_array = force_wav_data_to_mono(wav_array)
    return Sound(wav_array, sampling_freq)

  @classmethod
  def sound_from_gen_config(cls, sound_gen_config):
    if sound_gen_config.HasField('pure_tone_config'):
      time_series = gen_pure_tone_time_series(sound_gen_config)
      return Sound(time_series, sound_gen_config.fs)
    elif sound_gen_config.HasField('flat_spectrum_noise_config'):
      time_series = gen_flat_spectrum_time_series(sound_gen_config)
      return Sound(time_series, sound_gen_config.fs)

  @classmethod
  def sound_from_gen_config_path(cls, sound_gen_config_path):
    with open(sound_gen_config_path, 'r') as f:
      sound_gen_config = text_format.Parse(f.read(), sound_generation_pb2)
    return Sound.sound_from_config(sound_gen_config)

  def compute_spectrogram(self, **kwargs):
    return sig.spectrogram(self._time_series, self._sampling_freq, **kwargs)

  def _get_sliding_window_bounds(self, win_duration_ms, step_ms):
    bounds = []
    samples_per_window = int(self._sampling_freq*(win_duration_ms/1000))
    samples_per_step = int(self._sampling_freq * (step_ms / 1000))
    for start in range(0, len(self._time_series), samples_per_step):
      bounds.append(
        (start, min([start+samples_per_window, len(self._time_series)])))
    return bounds

  def _get_band_filtered_ts(self, start_freq, stop_freq, filter_order):
    return _filter_band(self._time_series, self._sampling_freq, start_freq,
                        stop_freq, filter_order)

  def _decompose_to_freq_bands(self, band_limits, filter_order):
    # TODO(kane): convert to name tuple
    signals = {}
    for start_freq, stop_freq in band_limits:
      assert(start_freq < stop_freq)
      assert(stop_freq <= self._sampling_freq/2)
      signals[(start_freq, stop_freq)] = self._get_band_filtered_ts(
        start_freq, stop_freq, filter_order)
    return signals

  # TODO(kane): This currently assumes underwater (ref 1uPa)
  def _get_windowed_spl(self, signal, win_duration_ms, step_ms):
    bounds = self._get_sliding_window_bounds(win_duration_ms, step_ms)
    spls = []
    for start, stop in bounds:
      segment = signal[start:stop]
      rms = np.sqrt(np.mean(segment**2))
      spls.append(20*np.log10(rms))
    return spls, bounds

  def get_windowed_spl_by_bands(self, band_limits, filter_order,
                                win_duration_ms, step_ms):
    signals = self._decompose_to_freq_bands(band_limits, filter_order)
    spls = {}
    for band, signal in signals.items():
      spls[band] = self._get_windowed_spl(signal, win_duration_ms, step_ms)
    return spls, signals

  # TODO(kane): Add support for plot configuration
  def plot_spectrogram(self, **kwargs):
    f, t, Sxx = self.compute_spectrogram(**kwargs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim(0, self._sampling_freq / 2)
    plt.show()

  # TODO(kane): Add support for plot configuration
  def plot_time_series(self):
    plt.plot(self._times, self._time_series)
    plt.ylabel('Amplitude')
    plt.xlabel('Time [sec]')
    plt.show()


class MaskingAnalyzer:
  def __init__(self, signal: Sound, noise: Sound):
    self.signal = signal
    self.noise = noise
