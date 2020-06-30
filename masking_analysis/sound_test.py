import os
from masking_analysis.protos import sound_pb2, sound_generation_pb2
from masking_analysis.sound import Sound
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

  def test_compute_spectrogram(self):
    # TODO(kane): Test contents of spectrogram
    self.test_sound.compute_spectrogram()


if __name__ == '__main__':
  unittest.main()
