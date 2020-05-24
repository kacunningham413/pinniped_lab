import os
from sound import Sound
import unittest


class SoundTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
