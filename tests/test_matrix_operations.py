import numpy as np
import pytest_funcnodes

from funcnodes_opencv.image_operations.matrix_operations import (
    merge,
    split,
    transpose,
    repeat,
    getAffineTransform,
    getPerspectiveTransform,
    getPerspectiveTransform_points,
    getAffineTransform_points,
)


@pytest_funcnodes.nodetest(merge)
async def test_merge(image1, image2):
    a = image1.data[:, :, 0]
    b = image2.data[:, :, 0]
    c = image1.data[:, :, 1] if image1.testchannels == 3 else image1.data[:, :, 0]
    result = (await merge.inti_call(channels=[a, b, c])).data
    assert result.shape == (image1.height(), image1.width(), 3)
    # showdat([image1, image2], result)
    np.testing.assert_array_equal(result[:, :, 0], a)
    np.testing.assert_array_equal(result[:, :, 1], b)
    np.testing.assert_array_equal(result[:, :, 2], c)


@pytest_funcnodes.nodetest(split)
async def test_split(image1):
    result = await split.inti_call(img=image1)
    assert len(result) == image1.testchannels
    np.testing.assert_array_equal((result[0].data)[:, :, 0], image1.data[:, :, 0])
    if image1.testchannels == 3:
        np.testing.assert_array_equal((result[1].data)[:, :, 0], image1.data[:, :, 1])
        np.testing.assert_array_equal((result[2].data)[:, :, 0], image1.data[:, :, 2])


@pytest_funcnodes.nodetest(transpose)
async def test_transpose(image1):
    result = (await transpose.inti_call(img=image1)).data
    assert result.shape == (image1.width(), image1.height(), image1.testchannels)
    np.testing.assert_array_equal(result[:, :, 0], image1.data[:, :, 0].T)
    if image1.testchannels == 3:
        np.testing.assert_array_equal(result[:, :, 1], image1.data[:, :, 1].T)
        np.testing.assert_array_equal(result[:, :, 2], image1.data[:, :, 2].T)


@pytest_funcnodes.nodetest(repeat)
async def test_repeat(image1):
    result = (await repeat.inti_call(img=image1, ny=2, nx=3)).data
    h = image1.height()
    w = image1.width()
    assert result.shape == (
        h * 2,
        w * 3,
        image1.testchannels,
    )
    # showdat([image1], result)
    np.testing.assert_array_equal(result[:h, :w, :], image1.data)
    np.testing.assert_array_equal(result[h:, :w, :], image1.data)
    np.testing.assert_array_equal(result[:h, w : 2 * w, :], image1.data)
    np.testing.assert_array_equal(result[:h, 2 * w :, :], image1.data)


@pytest_funcnodes.nodetest(getAffineTransform)
async def test_getAffineTransform():
    result = await getAffineTransform.inti_call(
        src=np.array([[0, 0], [0, 100], [100, 0]]),
        dst=np.array([[0, 0], [0, 100], [100, 0]]),
    )
    assert result.shape == (2, 3)


@pytest_funcnodes.nodetest(getPerspectiveTransform)
async def test_getPerspectiveTransform():
    result = await getPerspectiveTransform.inti_call(
        src=np.array([[0, 0], [0, 100], [100, 0], [100, 100]]),
        dst=np.array([[0, 0], [0, 100], [100, 0], [100, 100]]),
    )
    assert result.shape == (3, 3)


@pytest_funcnodes.nodetest(getPerspectiveTransform_points)
async def test_getPerspectiveTransform_points():
    result = await getPerspectiveTransform_points.inti_call(
        i1x1=0,
        i1y1=0,
        i1x2=0,
        i1y2=100,
        i1x3=100,
        i1y3=0,
        i1x4=100,
        i1y4=100,
        i2x1=0,
        i2y1=0,
        i2x2=0,
        i2y2=100,
        i2x3=100,
        i2y3=0,
        i2x4=100,
        i2y4=100,
    )
    assert result.shape == (3, 3)


@pytest_funcnodes.nodetest(getAffineTransform_points)
async def test_getAffineTransform_points():
    result = await getAffineTransform_points.inti_call(
        i1x1=0,
        i1y1=0,
        i1x2=0,
        i1y2=100,
        i1x3=100,
        i1y3=0,
        i2x1=0,
        i2y1=0,
        i2x2=0,
        i2y2=100,
        i2x3=100,
        i2y3=0,
    )
    assert result.shape == (2, 3)
