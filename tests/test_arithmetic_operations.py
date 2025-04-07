import numpy as np
import cv2
import pytest_funcnodes
from funcnodes_opencv.utils import assert_opencvdata
from .testuilts import prep

from funcnodes_opencv.image_operations.arithmetic_operations import (
    add,
    subtract,
    multiply,
    divide,
    addWeighted,
    where,
    brighten,
)


# --- Tests for arithmetic nodes ---


@pytest_funcnodes.nodetest(add)
async def test_add(image1, image2):
    image1, image2 = prep(image1, image2)
    res = cv2.add(image1.raw_transformed, image2.raw_transformed)
    res = assert_opencvdata(res)
    fnout = (await add.inti_call(img1=image1, img2=image2)).data
    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(subtract)
async def test_subtract(image1, image2):
    image1, image2 = prep(image1, image2)
    res = cv2.subtract(image1.raw_transformed, image2.raw_transformed)
    res = assert_opencvdata(res)
    fnout = (await subtract.inti_call(img1=image1, img2=image2)).data
    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-7)


@pytest_funcnodes.nodetest(multiply)
async def test_multiply(image1, image2):
    image1, image2 = prep(image1, image2)
    res = cv2.multiply(image1.raw_transformed / 255.0, image2.raw_transformed / 255.0)
    res = assert_opencvdata(res)
    fnout = (await multiply.inti_call(img1=image1, img2=image2)).data
    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(divide)
async def test_divide(image1, image2):
    image1, image2 = prep(image1, image2)
    res = cv2.divide(
        image1.raw_transformed / 255.0, image2.raw_transformed / 255.0 + 1e-10
    )
    res = np.clip(res, 0, 1)

    res = assert_opencvdata(res)
    fnout = (await divide.inti_call(img1=image1, img2=image2)).data
    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(addWeighted)
async def test_addWeighted(image1, image2):
    image1, image2 = prep(image1, image2)
    res = cv2.addWeighted(
        image1.raw_transformed,
        0.5,
        image2.raw_transformed,
        0.5,
        0,
    )
    res = assert_opencvdata(res)
    fnout = (await addWeighted.inti_call(img1=image1, img2=image2)).data
    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=5e-3)


@pytest_funcnodes.nodetest(where)
async def test_where_with_int(image1, mask):
    (image1,) = prep(image1)

    res = image1.raw_transformed / 255
    maskd = np.squeeze(mask.data / 255)
    res[maskd != 0] = 0.5

    res = assert_opencvdata(res)
    fnout = (await where.inti_call(img=image1, mask=mask, value=0.5)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res)


@pytest_funcnodes.nodetest(where)
async def test_where_with_image(image1, mask, image2):
    (image1, image2) = prep(image1, image2)

    res = image1.raw_transformed / 255
    maskd = np.squeeze(mask.data / 255)
    res[maskd != 0] = image2.raw_transformed[maskd != 0] / 255

    res = assert_opencvdata(res)
    fnout = (await where.inti_call(img=image1, mask=mask, value=image2)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res)


@pytest_funcnodes.nodetest(brighten)
async def test_brighten(image1):
    res = image1.raw_transformed / 255 + 0.5
    res = np.clip(res, 0, 1)
    res = assert_opencvdata(res)
    fnout = (await brighten.inti_call(img=image1, value=0.5)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res)
