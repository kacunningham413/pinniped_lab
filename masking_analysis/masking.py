from __future__ import annotations

from google.protobuf import text_format
import logging
import matplotlib.pyplot as plt
import numpy as np

from .sound import Sound, FreqBand

from masking_analysis.protos import masking_config_pb2, experiment_config_pb2
from typing import MutableMapping


def masking_analyzer_from_exp_config_txt(path: str) -> MaskingAnalyzer:
  """Generates a MaskingAnalzer objext from an experiment_config text proto."""
  experiment = experiment_config_pb2.Experiment()
  with open(path, 'r') as f:
    text_format.Parse(f.read(), experiment)
  return MaskingAnalyzer.from_experiment_config(experiment)


def add_sounds(s1: Sound, s2: Sound):
  # For 1st version, restrict to already aligned sounds
  # TODO(kane): Automatically align misaligned time series.
  assert (s1.times == s2.times)
  summed_time_series = s1.time_series + s2.time_series
  return Sound(summed_time_series, s1.sampling_freq)


def get_padded_signal(signal: Sound, noise: Sound,
                      signal_position: float) -> Sound:
  signal_len = len(signal.time_series)
  noise_len = len(noise.time_series)
  left_pad_size = int(noise_len * signal_position)
  if (left_pad_size + signal_len) > noise_len:
    logging.warning("Signal extends beyond noise and will be truncated.")
    padded_signal = np.concatenate([np.zeros(left_pad_size), signal.time_series[
                                                             :noise_len - left_pad_size]])
  else:
    right_pad_size = noise_len - left_pad_size - signal_len
    padded_signal = np.concatenate(
      [np.zeros(left_pad_size), signal.time_series, np.zeros(
        right_pad_size)])
  return Sound(padded_signal, signal.sampling_freq)


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
    if signal.sampling_freq != noise.sampling_freq:
      raise ValueError("Signal and Noise must have equal sampling frequencies.")
    signal = get_padded_signal(signal, noise, experiment_config.signal_position)
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
      excesses = np.array(band_signal_spls) - (np.array(
        band_noise_spls) - 10 * np.log10(bandwidth)) - band.critical_ratio
      signal_excesses[FreqBand(band)] = excesses
    return signal_excesses

  def get_signal_excess_times(self):
    times = [x for x in
             range(0, int(self.noise.duration_ms),
                   self.masking_config.window_step_ms)]
    # Offset times to center of sliding window
    times = [x + self.masking_config.window_duration_ms / 2 for x in times]
    return times

  def plot_signal_and_noise_spectrogram(self):
    summed = add_sounds(self.signal, self.noise)
    summed.plot_spectrogram()

  def plot_signal_excesses(self):
    se = self.get_signal_excess()
    se_t = self.get_signal_excess_times()
    for band, excess in se.items():
      plt.plot(se_t, excess, label=f'{band.start_freq}-{band.stop_freq}Hz')
    plt.legend(loc='upper right')
