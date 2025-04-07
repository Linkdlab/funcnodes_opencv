import numpy as np
import cv2
import pytest_funcnodes
from funcnodes_opencv.utils import assert_opencvdata
from funcnodes_opencv.image_operations.bitwise_operations import (
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_not,
)
from .testuilts import prep


# --- Tests for bitwise nodes ---
@pytest_funcnodes.nodetest(bitwise_and)
async def test_bitwise_and(image1, image2, mask):
    # cv2.bitwise_and performs pixel-wise bitwise AND operation.
    (image1, image2) = prep(image1, image2)
    res = cv2.bitwise_and(
        image1.raw_transformed, image2.raw_transformed, mask=mask.mask_raw
    )
    res = assert_opencvdata(res)
    fnout = (await bitwise_and.inti_call(img1=image1, img2=image2, mask=mask)).data

    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(bitwise_or)
async def test_bitwise_or(image1, image2, mask):
    # cv2.bitwise_or performs pixel-wise bitwise OR operation.
    (image1, image2) = prep(image1, image2)
    res = cv2.bitwise_or(
        image1.raw_transformed, image2.raw_transformed, mask=mask.mask_raw
    )
    res = assert_opencvdata(res)
    fnout = (await bitwise_or.inti_call(img1=image1, img2=image2, mask=mask)).data

    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(bitwise_xor)
async def test_bitwise_xor(image1, image2, mask):
    # cv2.bitwise_xor performs pixel-wise bitwise XOR operation.
    (image1, image2) = prep(image1, image2)
    res = cv2.bitwise_xor(
        image1.raw_transformed, image2.raw_transformed, mask=mask.mask_raw
    )
    res = assert_opencvdata(res)
    fnout = (await bitwise_xor.inti_call(img1=image1, img2=image2, mask=mask)).data

    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(bitwise_not)
async def test_bitwise_not(image1, mask):
    # cv2.bitwise_not performs pixel-wise bitwise NOT operation.

    res = cv2.bitwise_not(image1.raw_transformed, mask=mask.mask_raw)
    res = assert_opencvdata(res)
    fnout = (await bitwise_not.inti_call(img=image1, mask=mask)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)
