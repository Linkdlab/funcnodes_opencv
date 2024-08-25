import funcnodes_opencv as fnocv
import unittest
import cv2


class TestImageOperations(unittest.TestCase):
    def test_interpolations(self):
        interp = fnocv.image_operations.Interpolations

        self.assertEqual(interp.NEAREST.value, cv2.INTER_NEAREST)
        self.assertEqual(interp.LINEAR.value, cv2.INTER_LINEAR)
        self.assertEqual(interp.CUBIC.value, cv2.INTER_CUBIC)
        self.assertEqual(interp.AREA.value, cv2.INTER_AREA)
        self.assertEqual(interp.LANCZOS4.value, cv2.INTER_LANCZOS4)

        self.assertEqual(interp.v(interp.NEAREST), cv2.INTER_NEAREST)
        self.assertEqual(interp.v("NEAREST"), cv2.INTER_NEAREST)
        self.assertEqual(interp.v(0), cv2.INTER_NEAREST)
