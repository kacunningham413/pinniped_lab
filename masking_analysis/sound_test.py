import os
from masking_analysis.protos import sound_pb2, sound_generation_pb2
from masking_analysis.sound import Sound
import numpy as np
import unittest


class SoundTest(unittest.TestCase):
  def setUp(self):
    test_times_series = [0, 1]
    test_sampling_frequency = 100
    self.test_sound = Sound(test_times_series, test_sampling_frequency)

  def test_constructor(self):
    test_times_series = [0, 1]
    test_sampling_frequency = 100
    test_sound = Sound(test_times_series, test_sampling_frequency)
    self.assertTrue((test_sound._time_series == test_times_series).all())
    self.assertEqual(test_sound._sampling_freq, test_sampling_frequency)

  def test_sound_from_wav(self):
    root = os.path.abspath(os.path.dirname(__file__))
    test_path = os.path.join(root, "../test_data/sample.wav")
    test_sound = Sound.sound_from_wav(test_path)
    self.assertEqual(test_sound._sampling_freq, 8000)

  def test_pure_tone_from_gen_config(self):
    sound_gen_config = sound_generation_pb2.SoundGenConfig()
    sound_gen_config.fs = 10000
    sound_gen_config.duration = 1
    sound_gen_config.pure_tone_config.center_freq = 100
    sound = Sound.sound_from_gen_config(sound_gen_config)
    self.assertEqual(len(sound._time_series),
                     sound_gen_config.fs * sound_gen_config.duration)
    self.assertAlmostEqual(sound._time_series[0], 0)
    self.assertAlmostEqual(sound._time_series[int(sound_gen_config.fs / (
          sound_gen_config.pure_tone_config.center_freq * 4))], 1, 5)
    self.assertAlmostEqual(sound._time_series[int(sound_gen_config.fs / (
        sound_gen_config.pure_tone_config.center_freq * (4/3)))], -1, 5)

  def test_flat_spectrum_noise_from_gen_config(self):
    sound_gen_config = sound_generation_pb2.SoundGenConfig()
    sound_gen_config.fs = 2000
    sound_gen_config.duration = 1
    start_freq = 400
    stop_freq = 600
    sound_gen_config.flat_spectrum_noise_config.start_freq = start_freq
    sound_gen_config.flat_spectrum_noise_config.stop_freq = stop_freq
    sound_gen_config.flat_spectrum_noise_config.filter_order = 10
    sound = Sound.sound_from_gen_config(sound_gen_config)
    self.assertEqual(len(sound._time_series),
                     sound_gen_config.fs * sound_gen_config.duration)
    f, t, Sxx = sound.compute_spectrogram(nperseg=sound.time_series.size)
    Sx = np.squeeze(Sxx)
    Sx = Sx/np.sum(Sx)
    sum_in_band = 0
    sum_out_band = 0
    for f, Sx in zip(f, Sx):
      if start_freq <= f <= stop_freq:
        sum_in_band += Sx
      else:
        sum_out_band += Sx
    self.assertGreater(sum_in_band, 0.95)
    self.assertLess(sum_out_band, 0.05)

  def test_get_windowed_spl_by_bands(self):
    sound_gen_config = sound_generation_pb2.SoundGenConfig()
    sound_gen_config.fs = 10000
    sound_gen_config.duration = 1
    sound_gen_config.pure_tone_config.center_freq = 1000
    sound = Sound.sound_from_gen_config(sound_gen_config)
    band_limits = [(100, 200), (900, 1100), (3000, 4000)]
    filter_order = 10
    win_duration_ms = 250
    step_ms = 250
    spls, signals = sound.get_windowed_spl_by_bands(band_limits, filter_order,
                                                    win_duration_ms, step_ms)
    self.assertEqual(len(spls), len(signals))
    self.assertEqual(band_limits,
                     [(start, stop) for start, stop in spls.keys()])
    self.assertEqual(len(list(spls.values())[0][0]),
                     sound_gen_config.duration*1000/step_ms)
    # RMS of sin with amplitude 1 ~0.707, 20*log(.707) = -3.1 dB
    self.assertAlmostEqual(np.mean(spls[(900, 1100)][0]), -3.1, 1)
    # Assert that SPL in bands not containing signal is less than -100 dB
    self.assertLess(np.mean(spls[(100, 200)][0]), -100)
    self.assertLess(np.mean(spls[(3000, 4000)][0]), -100)

  def test_compute_spectrogram(self):
    # TODO(kane): Test contents of spectrogram
    self.test_sound.compute_spectrogram()


if __name__ == '__main__':
  unittest.main()
