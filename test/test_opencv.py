import unittest
import numpy as np
import cv2
from funcnodes_opencv import OpenCVImageFormat
from funcnodes_images import NumpyImageFormat, PillowImageFormat
from PIL import Image


class TestOpenCVImageFormat(unittest.TestCase):
    def setUp(self):
        # Create a sample OpenCV image
        self.sample_data = np.random.randint(0, 255, (100, 50, 3), dtype=np.uint8)
        if self.sample_data is not None:
            self.sample_data = cv2.cvtColor(
                self.sample_data, cv2.COLOR_RGB2BGR
            )  # Converting to BGR if necessary

    def test_initialization(self):
        """Test that the OpenCVImageFormat can be initialized properly."""
        self.assertIsInstance(OpenCVImageFormat(self.sample_data), OpenCVImageFormat)

    def test_color_conversion(self):
        """Test color space conversions within the OpenCVImageFormat."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        rgb_data = cv2_format.to_colorspace("RGB")
        self.assertEqual(rgb_data.shape, self.sample_data.shape)
        # Check if the color conversion was correctly applied
        b, g, r = cv2.split(self.sample_data)
        rgb_manual = cv2.merge([r, g, b])
        np.testing.assert_array_equal(rgb_data, rgb_manual)

    def test_jpeg_conversion(self):
        """Test JPEG conversion functionality."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        jpeg_data = cv2_format.to_jpeg()
        self.assertIsInstance(jpeg_data, bytes)

    def test_thumbnail_creation(self):
        """Test thumbnail creation functionality."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        thumbnail = cv2_format.to_thumbnail((50, 50))
        self.assertEqual(thumbnail.data.shape[1], 25)  # Checking width of the thumbnail
        self.assertEqual(thumbnail.data.shape[0], 50)  # Checking width of the thumbnail

    def test_resize(self):
        """Test resizing functionality."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        resized = cv2_format.resize(w=100)
        self.assertEqual(resized.data.shape[1], 100)  # Checking new width

    def test_to_numpy(self):
        """Test the conversion between cv2 and numpy formats."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        np_format = cv2_format.to_np()
        self.assertIsInstance(np_format, NumpyImageFormat)

        # Conversion back to cv2
        cv2_format_again = np_format.to_cv2()
        self.assertIsInstance(cv2_format_again, OpenCVImageFormat)

        np.testing.assert_array_equal(cv2_format_again.data, self.sample_data)

    def test_to_img(self):
        """Test the conversion to PIL image."""
        cv2_format = OpenCVImageFormat(self.sample_data)
        pil_img = cv2_format.to_img()
        self.assertIsInstance(pil_img, PillowImageFormat)
        self.assertEqual(np.prod(pil_img.data.size) * 3, cv2_format.data.size)

    def test_from_numpy(self):
        """Test the conversion from numpy to cv2."""
        np_format = NumpyImageFormat(self.sample_data)
        cv2_format = np_format.to_cv2()
        self.assertIsInstance(cv2_format, OpenCVImageFormat)
        np.testing.assert_array_equal(
            cv2_format.data,
            self.sample_data[:, :, [2, 1, 0]],  # Converting to BGR
        )

    def test_from_img(self):
        """Test the conversion from PIL image to cv2."""
        pil_img = PillowImageFormat(Image.fromarray(self.sample_data))
        cv2_format = pil_img.to_cv2()
        self.assertIsInstance(cv2_format, OpenCVImageFormat)
        np.testing.assert_array_equal(
            cv2_format.data,
            self.sample_data[:, :, [2, 1, 0]],  # Converting to BGR
        )
