import cv2
from funcnodes_opencv.utils import assert_opencvdata, assert_similar_opencvdata
from funcnodes_opencv.imageformat import (
    OpenCVImageFormat,
)
import numpy as np
from .testuilts import prep


def test_testdata(image1, image2):
    # test if the image is a numpy array
    assert isinstance(image1, OpenCVImageFormat)
    assert isinstance(image2, OpenCVImageFormat)

    # test if the image is a numpy array
    assert isinstance(image1.data, np.ndarray)
    assert isinstance(image2.data, np.ndarray)

    # test if the image is a numpy array
    assert image1.data.shape[2] == image1.testchannels
    assert image2.data.shape[2] == image2.testchannels

    if image1.testchannels == 1:
        assert image1.raw_transformed.ndim == 2
    else:
        assert image1.raw_transformed.shape[2] == image1.testchannels

    if image2.testchannels == 1:
        assert image2.raw_transformed.ndim == 2
    else:
        assert image2.raw_transformed.shape[2] == image2.testchannels


def test_imagedata(image1, image1_raw):
    if image1.testchannels == 1:
        image1_raw = cv2.cvtColor(image1_raw, cv2.COLOR_BGR2GRAY)

    i1, i2 = assert_similar_opencvdata(
        image1.data, assert_opencvdata(image1_raw, channel=image1.testchannels)
    )
    if image1.testdtype in [bool, np.bool]:
        # shared between uint8 and bool is uint 8 but we want to compare bool values
        i2 = i2.astype(bool).astype(np.uint8) * 255

    np.testing.assert_array_equal(i1, i2)


def test_prep(
    image1,
    image2,
):
    image1, image2 = prep(image1, image2)
    d1, d2, i1, i2 = assert_similar_opencvdata(
        image1, image2, image1.raw_transformed, image2.raw_transformed
    )
    np.testing.assert_allclose(i1.astype(float), d1.astype(float))
    np.testing.assert_allclose(i2.astype(float), d2.astype(float))
