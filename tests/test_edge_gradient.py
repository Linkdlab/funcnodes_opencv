import numpy as np
import cv2
import pytest
import pytest_funcnodes

from funcnodes_opencv.image_processing.edge_gradient import (
    Canny,
    Laplacian,
    Sobel,
    Scharr,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(Canny)
async def test_Canny(image1):
    res = cv2.Canny(image1.raw_transformed, 100, 200)
    res = assert_opencvdata(res)
    fnout = (await Canny.inti_call(img=image1)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest.mark.parametrize(
    "dx, dy, ksize",
    [
        (1, 0, 3),
        (0, 1, 3),
        (1, 1, 3),
        (1, 0, 5),
        (0, 1, 5),
        (1, 1, 5),
    ],
)
@pytest_funcnodes.nodetest(Sobel)
async def test_Sobel(image1, dx, dy, ksize):
    res = cv2.Sobel(image1.raw_transformed, -1, dx, dy, ksize=ksize)
    res = assert_opencvdata(res)
    fnout = (await Sobel.inti_call(img=image1, dx=dx, dy=dy, ksize=ksize)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "dx, dy",
    [
        (1, 0),
        (0, 1),
    ],
)
@pytest_funcnodes.nodetest(Scharr)
async def test_Scharr(image1, dx, dy):
    res = cv2.Scharr(image1.raw_transformed, -1, dx, dy)
    res = assert_opencvdata(res)
    fnout = (await Scharr.inti_call(img=image1, dx=dx, dy=dy)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "ksize",
    [
        1,
        3,
        5,
    ],
)
@pytest_funcnodes.nodetest(Laplacian)
async def test_Laplacian(image1, ksize):
    res = cv2.Laplacian(image1.raw_transformed, -1, ksize=ksize)
    res = assert_opencvdata(res)
    fnout = (await Laplacian.inti_call(img=image1, ksize=ksize)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-5)
