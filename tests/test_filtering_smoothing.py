import numpy as np
import cv2
import pytest
import pytest_funcnodes
from funcnodes_opencv.image_processing.filtering_smoothing import (
    blur,
    gaussianBlur,
    medianBlur,
    bilateralFilter,
    stackBlur,
    boxFilter,
    filter2D,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest.mark.parametrize(
    "kw",
    np.arange(1, 50, 10),
)
@pytest.mark.parametrize(
    "kh",
    np.arange(0, 50, 10),
)
@pytest_funcnodes.nodetest(blur)
async def test_blur(image1, kw, kh):
    res = cv2.blur(image1.raw_transformed, (kw, kh if kh > 0 else kw))
    res = assert_opencvdata(res)
    fnout = (await blur.inti_call(img=image1, kw=kw, kh=kh)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-3)


@pytest.mark.parametrize(
    "kw",
    np.arange(1, 50, 10),
)
@pytest.mark.parametrize(
    "kh",
    np.arange(0, 50, 10),
)
@pytest_funcnodes.nodetest(gaussianBlur)
async def test_gaussianBlur(image1, kw, kh):
    _kh = kh
    _kw = kw
    if _kh <= 0:
        _kh = _kw

    if _kw % 2 == 0:
        _kw += 1

    if _kh % 2 == 0:
        _kh += 1

    res = cv2.GaussianBlur(image1.raw_transformed, (_kw, _kh), 0)
    res = assert_opencvdata(res)
    fnout = (await gaussianBlur.inti_call(img=image1, kw=kw, kh=kh)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-1, atol=5e-3)


@pytest.mark.parametrize(
    "ksize",
    np.arange(1, 51, 10),
)
@pytest_funcnodes.nodetest(medianBlur)
async def test_medianBlur(image1, ksize):
    res = cv2.medianBlur(image1.raw_transformed, ksize if ksize % 2 != 0 else ksize + 1)
    res = assert_opencvdata(res)
    fnout = (await medianBlur.inti_call(img=image1, ksize=ksize)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize(
    "d",
    np.arange(1, 22, 5),
)
@pytest.mark.parametrize(
    "sigmaColor",
    np.arange(0, 10, 2),
)
@pytest.mark.parametrize(
    "sigmaSpace",
    np.arange(0, 10, 2),
)
@pytest_funcnodes.nodetest(bilateralFilter)
async def test_bilateralFilter(image1, d, sigmaColor, sigmaSpace):
    # uint8img = image1.raw_transformed
    # uint8res = cv2.bilateralFilter(
    #     uint8img, d, int(sigmaColor * 255) + 1, int(sigmaSpace * 255) + 1
    # )
    # float32img = uint8img.astype(np.float32) / 255.0
    # float32res = cv2.bilateralFilter(float32img, d, sigmaColor, sigmaSpace)

    # cv2.imshow("uint8res", uint8res)
    # cv2.imshow("float32res", float32res)
    # cv2.imshow("uint8img", uint8img)
    # cv2.imshow("float32img", float32img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    res = cv2.bilateralFilter(
        image1.raw_transformed.astype(np.float32) / 255,
        d,
        sigmaColor,
        sigmaSpace,
    )
    res = assert_opencvdata(res)
    fnout = (
        await bilateralFilter.inti_call(
            img=image1, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=5e-3)


@pytest.mark.parametrize(
    "kw",
    np.arange(1, 5),
)
@pytest.mark.parametrize(
    "kh",
    np.arange(0, 5),
)
@pytest_funcnodes.nodetest(stackBlur)
async def test_stackBlur(image1, kw, kh):
    _kw = kw
    _kh = kh
    if _kw % 2 == 0:
        _kw += 1
    if _kh % 2 == 0:
        _kh += 1
    if _kh <= 0:
        _kh = _kw

    res = cv2.stackBlur(
        image1.raw_transformed,
        (
            _kw,
            _kh,
        ),
    )
    res = assert_opencvdata(res)
    fnout = (await stackBlur.inti_call(img=image1, kw=kw, kh=kh)).data
    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=9e-3)


@pytest.mark.parametrize(
    "kw",
    np.arange(1, 5),
)
@pytest.mark.parametrize(
    "kh",
    np.arange(1, 5),
)
@pytest_funcnodes.nodetest(boxFilter)
async def test_boxFilter(image1, kw, kh):
    res = cv2.boxFilter(
        image1.raw_transformed,
        -1,
        (kw, kh),
    )
    res = assert_opencvdata(res)
    fnout = (await boxFilter.inti_call(img=image1, kw=kw, kh=kh)).data
    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=4e-3)


@pytest_funcnodes.nodetest(filter2D)
async def test_filter2D(image1):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    res = cv2.filter2D(image1.raw_transformed, -1, kernel)
    res = assert_opencvdata(res)

    fnout = (await filter2D.inti_call(img=image1, kernel=kernel)).data

    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1e-5)
