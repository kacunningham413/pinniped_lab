from __future__ import annotations

from google.protobuf import text_format
import matplotlib.pyplot as plt
import numpy as np
import wave

from scipy import signal as sig
from masking_analysis.protos import sound_generation_pb2, masking_config_pb2
import sys
from typing import MutableMapping, Text, Tuple, Sequence


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


def gen_chirp_time_series(
    sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sequence[float]:
  x = np.linspace(0, sound_gen_config.duration,
                  sound_gen_config.duration * sound_gen_config.fs)
  sweep_methods = {
    sound_gen_config.chirp_config.LINEAR: 'linear',
    sound_gen_config.chirp_config.QUADRATIC: 'quadratic',
    sound_gen_config.chirp_config.LOGARITHMIC: 'logarithmic',
    sound_gen_config.chirp_config.HYPERBOLIC: 'hyperbolic',
  }
  ts = sig.chirp(x, sound_gen_config.chirp_config.start_freq,
                 sound_gen_config.duration,
                 sound_gen_config.chirp_config.stop_freq,
                 method=sweep_methods[
                   sound_gen_config.chirp_config.sweep_method])
  # TODO: Added this windowing here to help when the signal is shorter than
  #  noise and position is set. Should move windowing to sound gen config and
  #  remove from the signal excess calculation.
  ts = ts * sig.windows.tukey(len(ts))
  return ts


def gen_flat_spectrum_time_series(
    sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sequence[float]:
  white_noise = [np.random.normal() for _ in
                 range(sound_gen_config.duration * sound_gen_config.fs)]
  return _filter_band(white_noise, sound_gen_config.fs,
                      sound_gen_config.flat_spectrum_noise_config.start_freq,
                      sound_gen_config.flat_spectrum_noise_config.stop_freq,
                      sound_gen_config.flat_spectrum_noise_config.filter_order)


def scale_sound_by_spl(sound: Sound, target_spl: int,
                       tolerance: float = 0.1) -> Sound:
  """Returns a copy of the input sound scaled to target SPL.

  Target spl is for the entire duration and frequency range.

  Args:
    sound: Input sound to scale.
    target_spl: Target spl to scale to in dB re 1 uPa.
    tolerance: Resulting spl will be within this tolerance.
  """

  def _bin_search(lower: float, upper: float) -> float:
    mid = (lower + upper) / 2
    mid_spl = Sound(sound.time_series * mid, sound.sampling_freq).get_spl()
    if np.abs(mid_spl - target_spl) < tolerance:
      return mid
    elif mid_spl < target_spl:
      return _bin_search(mid, upper)
    elif mid_spl > target_spl:
      return _bin_search(lower, mid)

  max_scale = 1
  min_scale = 1
  spl = sound.get_spl()
  while spl > target_spl:
    min_scale /= 2
    spl = Sound(sound.time_series * min_scale, sound.sampling_freq).get_spl()
  spl = sound.get_spl()
  while spl < target_spl:
    max_scale = max_scale * 2
    spl = Sound(sound.time_series * max_scale, sound.sampling_freq).get_spl()
  scale_factor = _bin_search(min_scale, max_scale)
  return Sound(sound.time_series * scale_factor, sound.sampling_freq)


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

  @property
  def times(self):
    return self._times

  @property
  def sampling_freq(self):
    return self._sampling_freq

  @property
  def duration_ms(self):
    return len(self._times) / self.sampling_freq * 1000

  @classmethod
  def sound_from_wav(cls, wav_path: str) -> Sound:
    sampling_freq, wav_array = read_wav_file(wav_path)
    wav_array = force_wav_data_to_mono(wav_array)
    return Sound(wav_array, sampling_freq)

  @classmethod
  def sound_from_gen_config(
      cls, sound_gen_config: sound_generation_pb2.SoundGenConfig) -> Sound:
    if sound_gen_config.HasField('wavfile_config'):
      sound = Sound.sound_from_wav(sound_gen_config.wavfile_config.wav_path)
    elif sound_gen_config.HasField('pure_tone_config'):
      time_series = gen_pure_tone_time_series(sound_gen_config)
      sound = Sound(time_series, sound_gen_config.fs)
    elif sound_gen_config.HasField('flat_spectrum_noise_config'):
      time_series = gen_flat_spectrum_time_series(sound_gen_config)
      sound = Sound(time_series, sound_gen_config.fs)
    elif sound_gen_config.HasField('chirp_config'):
      time_series = gen_chirp_time_series(sound_gen_config)
      sound = Sound(time_series, sound_gen_config.fs)
    else:
      raise ValueError("Unrecognized Sound Generation Config type.")
    if sound_gen_config.HasField('calibration'):
      sound = scale_sound_by_spl(sound, sound_gen_config.calibration.spl)
    return sound

  @classmethod
  def sound_from_gen_config_path(cls, sound_gen_config_path: str) -> Sound:
    sound_gen_config = sound_generation_pb2.SoundGenConfig()
    with open(sound_gen_config_path, 'r') as f:
      text_format.Parse(f.read(), sound_generation_pb2)
    return Sound.sound_from_gen_config(sound_gen_config)

  def get_spl(self) -> float:
    """Returns sound pressure level re 1uPa."""
    mean_of_squares = np.mean(self._time_series ** 2)
    # Deal with special cases like zero-mean noise
    if mean_of_squares < 10 * sys.float_info.min:
      return 0
    rms = np.sqrt(mean_of_squares)
    return 20 * np.log10(rms)

  def compute_spectrogram(self, **kwargs):
    return sig.spectrogram(self._time_series, self._sampling_freq, **kwargs)

  def _get_sliding_window_bounds(self,
                                 win_duration_ms: int,
                                 step_ms: int) -> Sequence[TimeSlice]:
    bounds = []
    samples_per_window = int(self._sampling_freq * (win_duration_ms / 1000))
    samples_per_step = int(self._sampling_freq * (step_ms / 1000))
    for start in range(0, len(self._time_series) - 1, samples_per_step):
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
      spls[time_slice] = 20 * np.log10(rms + 10 ** -10)
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
