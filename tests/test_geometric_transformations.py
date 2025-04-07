import numpy as np
import cv2
import pytest
import pytest_funcnodes
from funcnodes_opencv.image_processing.geometric_transformations import (
    flip,
    rotate,
    resize,
    warpAffine,
    perpectiveTransform,
    freeRotation,
    pyrDown,
    pyrUp,
    FreeRotationCropMode,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(flip)
async def test_flip(image1):
    res = cv2.flip(image1.raw_transformed, 1)
    res = assert_opencvdata(res)

    fnout = (await flip.inti_call(img=image1)).data

    # showdat([image1], res, fnout)
    np.testing.assert_array_equal(
        fnout,
        res,
    )


@pytest_funcnodes.nodetest(rotate)
async def test_rotate(image1):
    res = cv2.rotate(image1.raw_transformed, cv2.ROTATE_90_CLOCKWISE)
    res = assert_opencvdata(res)

    fnout = (await rotate.inti_call(img=image1)).data

    # showdat([image1], res, fnout)
    np.testing.assert_array_equal(
        fnout,
        res,
    )


@pytest_funcnodes.nodetest(resize)
async def test_resize(image1):
    res = cv2.resize(image1.raw_transformed, (100, 200))
    res = assert_opencvdata(res)

    fnout = (await resize.inti_call(img=image1, h=200, w=100)).data
    # showdat([image1], res, fnout)

    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=3e-3,
    )


@pytest_funcnodes.nodetest(warpAffine)
async def test_warpAffine(image1):
    M = cv2.getRotationMatrix2D((50, 50), 45, 1)

    res = cv2.warpAffine(image1.raw_transformed, M, (100, 100))
    res = assert_opencvdata(res)

    fnout = (await warpAffine.inti_call(img=image1, M=M, h=100, w=100)).data
    # showdat([image1], res, fnout)

    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=3e-3,
    )


@pytest_funcnodes.nodetest(perpectiveTransform)
async def test_perpectiveTransform(image1):
    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32),
        np.array([[0, 0], [0, 100], [100, 0], [100, 100]], dtype=np.float32),
    )

    res = cv2.warpPerspective(image1.raw_transformed, M, (100, 100))
    res = assert_opencvdata(res)

    fnout = (await perpectiveTransform.inti_call(img=image1, M=M, h=100, w=100)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=3e-3,
    )


@pytest_funcnodes.nodetest(freeRotation)
@pytest.mark.parametrize(
    "crop_mode",
    [FreeRotationCropMode.CROP, FreeRotationCropMode.KEEP, FreeRotationCropMode.NONE],
)
async def test_freeRotation(image1, crop_mode):
    results = await freeRotation.inti_call(img=image1, angle=45, mode=crop_mode)
    arr, M = (results[0]).data, results[1]
    h, w = image1.raw_transformed.shape[:2]
    # new_width' = |w cos(θ)| + |h sin(θ)|
    # new_height' = |w sin(θ)| + |h cos(θ)|
    if crop_mode == FreeRotationCropMode.NONE:
        assert arr.shape == (
            image1.raw_transformed.shape[0],
            image1.raw_transformed.shape[1],
            image1.testchannels,
        )
    elif crop_mode == FreeRotationCropMode.KEEP:
        new_width = int(
            abs(w * np.cos(np.radians(45))) + abs(h * np.sin(np.radians(45)))
        )
        new_height = int(
            abs(w * np.sin(np.radians(45))) + abs(h * np.cos(np.radians(45)))
        )
        assert arr.shape == (
            new_height,
            new_width,
            image1.testchannels,
        )
    elif crop_mode == FreeRotationCropMode.CROP:
        assert arr.shape == (
            453,
            453,
            image1.testchannels,
        )
    else:
        raise ValueError("Invalid crop mode")

    res = cv2.warpAffine(image1.raw_transformed, M, arr.shape[:2][::-1])
    res = assert_opencvdata(res)

    # showdat([image1], res, arr)
    np.testing.assert_allclose(
        arr,
        res,
        rtol=1e-6,
        atol=3e-3,
    )


@pytest_funcnodes.nodetest(pyrDown)
async def test_pyrDown(image1):
    res = cv2.pyrDown(image1.raw_transformed)
    res = assert_opencvdata(res)

    fnout = (await pyrDown.inti_call(img=image1)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=3e-3,
    )


@pytest_funcnodes.nodetest(pyrUp)
async def test_pyrUp(image1):
    res = cv2.pyrUp(image1.raw_transformed)
    res = assert_opencvdata(res)

    fnout = (await pyrUp.inti_call(img=image1)).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=3e-3,
    )
