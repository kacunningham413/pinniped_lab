from __future__ import annotations

from google.protobuf import text_format
import matplotlib.pyplot as plt
import numpy as np
import wave

from scipy import signal as sig
from masking_analysis.protos import sound_generation_pb2, \
  masking_config_pb2, experiment_config_pb2
import sys
from typing import Mapping, MutableMapping, Text, Tuple, Sequence


def masking_analyzer_from_exp_config_txt(path: str) -> MaskingAnalyzer:
  """Generates a MaskingAnalzer objext from an experiment_config text proto."""
  experiment = experiment_config_pb2.Experiment()
  with open(path, 'r') as f:
    text_format.Parse(f.read(), experiment)
  return MaskingAnalyzer.from_experiment_config(experiment)


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


def force_wav_data_to_mono(wav_data: np.ndarray) -> np.ndarray:
  """Converts two channel wav data to a single channel."""
  ndims = len(wav_data.shape)
  if ndims == 1:
    return wav_data
  elif ndims == 2:
    return wav_data[0, :]
  else:
    raise ValueError("Wav file  is neither mono nor stereo")


def _filter_band(time_series: Sequence[float], fs: int, start_freq: int,
                 stop_freq: int, filter_order: int,
                 window: Sequence[int] = None) -> Sequence[float]:
  """Utility function for bandpass filtering a time series."""
  if window is not None:
    time_series = time_series * window
  nyquist_freq = fs / 2
  start_freq_ratio = start_freq / nyquist_freq
  stop_freq_ratio = stop_freq / nyquist_freq
  sos = sig.butter(filter_order, [start_freq_ratio, stop_freq_ratio],
                   btype='bandpass', output='sos')
  return sig.sosfilt(sos, time_series)


def gen_pure_tone_time_series(
    sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sequence[float]:
  x = np.arange(
    sound_gen_config.duration * sound_gen_config.fs) / sound_gen_config.fs
  return np.sin(2 * np.pi * sound_gen_config.pure_tone_config.center_freq * x)


def gen_flat_spectrum_time_series(
    sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sequence[float]:
  white_noise = [np.random.normal() for _ in
                 range(sound_gen_config.duration * sound_gen_config.fs)]
  return _filter_band(white_noise, sound_gen_config.fs,
                      sound_gen_config.flat_spectrum_noise_config.start_freq,
                      sound_gen_config.flat_spectrum_noise_config.stop_freq,
                      sound_gen_config.flat_spectrum_noise_config.filter_order)


class FreqBand:
  """
  Helper data class for frequency band definition.

  FreqBand objects are practically identical to masking_config_pb2.SingleBand
  derived object expect they are (1) hashable, and (2) have a defined equaltiy
  operator."""
  def __init__(self, band: masking_config_pb2.SingleBand):
    self.start_freq = band.start_freq
    self.stop_freq = band.stop_freq

  @classmethod
  def from_limits(cls, start_freq: int, stop_freq: int):
    band = masking_config_pb2.SingleBand()
    band.start_freq = start_freq
    band.stop_freq = stop_freq
    return FreqBand(band)

  def __hash__(self):
    return hash((self.start_freq, self.stop_freq))

  def __eq__(self, other: FreqBand):
    if type(other) is type(self):
      return (self.start_freq == other.start_freq) and (
          self.stop_freq == other.stop_freq)


class TimeSlice:
  """
  Helper data class for a time window band.

  Hashable and comparable for equality."""
  def __init__(self, start: int, stop: int):
    self.start = start
    self.stop = stop

  def __hash__(self):
    return hash((self.start, self.stop))

  def __eq__(self, other: TimeSlice):
    if type(other) is type(self):
      return (self.start == other.start) and (self.stop == other.stop)


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
  def sound_from_wav(cls, wav_path: str) -> Sound:
    sampling_freq, wav_array = read_wav_file(wav_path)
    wav_array = force_wav_data_to_mono(wav_array)
    return Sound(wav_array, sampling_freq)

  @classmethod
  def sound_from_gen_config(
      cls, sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sound:
    if sound_gen_config.HasField('pure_tone_config'):
      time_series = gen_pure_tone_time_series(sound_gen_config)
      return Sound(time_series, sound_gen_config.fs)
    elif sound_gen_config.HasField('flat_spectrum_noise_config'):
      time_series = gen_flat_spectrum_time_series(sound_gen_config)
      return Sound(time_series, sound_gen_config.fs)

  @classmethod
  def sound_from_gen_config_path(cls, sound_gen_config_path: str) -> Sound:
    sound_gen_config = sound_generation_pb2.SoundGenConfig()
    with open(sound_gen_config_path, 'r') as f:
      text_format.Parse(f.read(), sound_generation_pb2)
    return Sound.sound_from_gen_config(sound_gen_config)

  def compute_spectrogram(self, **kwargs):
    return sig.spectrogram(self._time_series, self._sampling_freq, **kwargs)

  def _get_sliding_window_bounds(self,
                                 win_duration_ms: int,
                                 step_ms: int) -> Sequence[TimeSlice]:
    bounds = []
    samples_per_window = int(self._sampling_freq * (win_duration_ms / 1000))
    samples_per_step = int(self._sampling_freq * (step_ms / 1000))
    for start in range(0, len(self._time_series), samples_per_step):
      bounds.append(TimeSlice(
        start, min([start + samples_per_window, len(self._time_series)])))
    return bounds

  def _get_band_filtered_sound(self, start_freq: int,
                               stop_freq: int,
                               filter_order: int,
                               window: np.array(float)) -> Sound:
    ts = _filter_band(self._time_series, self._sampling_freq, start_freq,
                      stop_freq, filter_order, window)
    return Sound(ts, self._sampling_freq)

  def _decompose_to_freq_bands(
      self, auditory_band_config: masking_config_pb2.AuditoryBandConfig) -> \
      MutableMapping[FreqBand, Sound]:
    sounds: MutableMapping[FreqBand, Sound] = {}
    for band in auditory_band_config.auditory_bands:
      assert (band.start_freq < band.stop_freq)
      assert (band.stop_freq <= self._sampling_freq / 2)
      sounds[FreqBand(band)] = self._get_band_filtered_sound(
        band.start_freq, band.stop_freq, auditory_band_config.filter_order,
        sig.tukey(len(self._time_series), alpha=0.1))
    return sounds

  # TODO(kane): This currently assumes underwater (ref 1uPa)
  def _get_windowed_spl(self, win_duration_ms: int,
                        step_ms: int) -> MutableMapping[TimeSlice, int]:
    slices = self._get_sliding_window_bounds(win_duration_ms, step_ms)
    spls: MutableMapping[TimeSlice, int] = {}
    for time_slice in slices:
      segment = self._time_series[time_slice.start:time_slice.stop]
      rms = np.sqrt(np.mean(segment ** 2))
      spls[time_slice] = 20 * np.log10(rms)
    return spls

  def get_windowed_spl_by_bands(
      self, auditory_band_config: masking_config_pb2.AuditoryBandConfig,
      win_duration_ms: int,
      step_ms: int) -> MutableMapping[FreqBand, MutableMapping[TimeSlice, int]]:
    sounds = self._decompose_to_freq_bands(auditory_band_config)
    spls: MutableMapping[FreqBand, MutableMapping[TimeSlice, int]] = {}
    for band, sound in sounds.items():
      spls[band] = sound._get_windowed_spl(win_duration_ms, step_ms)
    return spls

  # TODO(kane): Add support for plot configuration
  def plot_spectrogram(self, **kwargs):
    f, t, Sxx = self.compute_spectrogram(**kwargs)
    plt.pcolormesh(t, f, Sxx, shading='auto')
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
  def __init__(self, signal: Sound, noise: Sound,
               masking_config: masking_config_pb2):
    self.signal = signal
    self.noise = noise
    self.masking_config = masking_config

  @classmethod
  def from_experiment_config(cls, experiment_config: experiment_config_pb2):
    signal = Sound.sound_from_gen_config(experiment_config.signal_gen_config)
    noise = Sound.sound_from_gen_config(experiment_config.noise_gen_config)
    masking_config = experiment_config.masking_config
    return MaskingAnalyzer(signal, noise, masking_config)

  def get_signal_excess(self):
    signal_spls = self.signal.get_windowed_spl_by_bands(
      self.masking_config.auditory_band_config,
      self.masking_config.window_duration_ms,
      self.masking_config.window_step_ms
    )
    noise_spls = self.noise.get_windowed_spl_by_bands(
      self.masking_config.auditory_band_config,
      self.masking_config.window_duration_ms,
      self.masking_config.window_step_ms
    )
    assert (signal_spls.keys() == noise_spls.keys())
    signal_excesses: MutableMapping[FreqBand, np.ndarray] = {}
    for band in self.masking_config.auditory_band_config.auditory_bands:
      band_signal_spls = list(signal_spls[FreqBand(band)].values())
      band_noise_spls = list(noise_spls[FreqBand(band)].values())
      bandwidth = band.stop_freq - band.start_freq
      excesses = (np.array(band_signal_spls) - np.array(
        band_noise_spls)) - 10 * np.log10(bandwidth) - band.critical_ratio
      signal_excesses[FreqBand(band)] = excesses
    return signal_excesses
