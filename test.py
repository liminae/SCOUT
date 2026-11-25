import unittest

import cv2
import numpy as np

from geometry import landmarks, profile
from vision import analyze, demo


class Tests(unittest.TestCase):
    def test_symmetric(self):
        result, overlay = analyze(demo())
        self.assertLess(result['centerline_rms'], .005)
        self.assertEqual(overlay.shape, (400, 320, 3))

    def test_bend(self):
        a, _ = analyze(demo())
        b, _ = analyze(demo(30))
        self.assertGreater(b['centerline_rms'], a['centerline_rms']+.04)

    def test_inversion(self):
        a, _ = analyze(demo(15))
        b, _ = analyze(255-demo(15))
        self.assertAlmostEqual(a['centerline_rms'], b['centerline_rms'], places=2)

    def test_empty(self):
        with self.assertRaises(ValueError):
            analyze(np.full((100, 100, 3), 255, np.uint8))

    def test_profile(self):
        with self.assertRaises(ValueError):
            profile(np.zeros((100, 100), np.uint8))

    def test_landmarks(self):
        p = [[.2, .2], [.8, .2], [.2, .8], [.8, .8]]
        self.assertEqual(landmarks(p, 320, 400), dict(upper_offset=0, lower_offset=0, center_offset=0))
        p[3][1] = .9
        self.assertAlmostEqual(landmarks(p, 320, 400)['lower_offset'], 40/192)

    def test_invalid_landmarks(self):
        with self.assertRaises(ValueError):
            landmarks([[0, 0]]*4)


if __name__=='__main__':
    unittest.main()
