from masking_analysis.protos import sound_generation_pb2, \
  masking_config_pb2
from masking_analysis.sound import Sound, FreqBand
from masking_analysis.masking import MaskingAnalyzer, masking_analyzer_from_exp_config_txt
import numpy as np
import unittest


class MaskingTest(unittest.TestCase):
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

  def test_masking_analyzer_from_config(self):
    analyzer = masking_analyzer_from_exp_config_txt(
      './test_data/experiment_config_example.textproto')
    self.assertIsInstance(analyzer, MaskingAnalyzer)


if __name__ == '__main__':
  unittest.main()
