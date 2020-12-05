import os
from masking_analysis.protos import sound_generation_pb2, \
  masking_config_pb2
from masking_analysis.sound import Sound, MaskingAnalyzer, FreqBand
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
        sound_gen_config.pure_tone_config.center_freq * (4 / 3)))], -1, 5)

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
    Sx = Sx / np.sum(Sx)
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
    sound_gen_config.duration = 3
    sound_gen_config.pure_tone_config.center_freq = 1000
    sound = Sound.sound_from_gen_config(sound_gen_config)
    band_config = masking_config_pb2.AuditoryBandConfig()
    for start, stop in [(100, 200), (900, 1100), (3000, 4000)]:
      band = band_config.auditory_bands.add()
      band.start_freq = start
      band.stop_freq = stop
    band_config.filter_order = 10
    win_duration_ms = 500
    step_ms = 500
    spls = sound.get_windowed_spl_by_bands(band_config,
                                           win_duration_ms, step_ms)
    self.assertEqual([FreqBand(band) for band in band_config.auditory_bands],
                     list(spls.keys()))
    self.assertEqual(len(list(spls.values())[0][0]),
                     sound_gen_config.duration * 1000 / step_ms)

    # RMS of sin with amplitude 1 ~0.707, 20*log(.707) = -3.1 dB, slightly less
    # when we apply a Tukey window before band filtering. We exclude extrema to
    # account for the window
    self.assertAlmostEqual(
      np.mean(spls[FreqBand.from_limits(900, 1100)][0][1:-1]), -3.0, 1)
    # Assert that SPL in bands not containing signal is less than -100 dB
    self.assertLess(np.mean(spls[FreqBand.from_limits(100, 200)][0]), -100)
    self.assertLess(np.mean(spls[FreqBand.from_limits(3000, 4000)][0]), -100)

  def test_masking_analyzer_signal_excess(self):
    signal_gen_config = sound_generation_pb2.SoundGenConfig()
    signal_gen_config.fs = 5000
    signal_gen_config.duration = 2
    signal_gen_config.pure_tone_config.center_freq = 500
    signal = Sound.sound_from_gen_config(signal_gen_config)

    noise_gen_config = sound_generation_pb2.SoundGenConfig()
    noise_gen_config.fs = 5000
    noise_gen_config.duration = 2
    noise_gen_config.flat_spectrum_noise_config.start_freq = 50
    noise_gen_config.flat_spectrum_noise_config.stop_freq = 1000
    noise_gen_config.flat_spectrum_noise_config.filter_order = 10
    noise = Sound.sound_from_gen_config(noise_gen_config)

    masking_config = masking_config_pb2.MaskingConfig()
    band_config = masking_config.auditory_band_config
    for start, stop in [(100, 200), (200, 400), (400, 800)]:
      band = band_config.auditory_bands.add()
      band.start_freq = start
      band.stop_freq = stop
    band_config.filter_order = 10
    masking_config.window_duration_ms = 250
    masking_config.window_step_ms = 250

    analyzer = MaskingAnalyzer(signal, noise, masking_config)
    se = analyzer.get_signal_excess()
    self.assertTrue(np.all((se[FreqBand.from_limits(400, 800)] - se[
      FreqBand.from_limits(100, 200)]) > 10))
    self.assertTrue(np.all((se[FreqBand.from_limits(400, 800)] - se[
      FreqBand.from_limits(200, 400)]) > 10))

  def test_compute_spectrogram(self):
    # TODO(kane): Test contents of spectrogram
    self.test_sound.compute_spectrogram()


if __name__ == '__main__':
  unittest.main()
