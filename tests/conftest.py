import numpy as np
import pytest
import cv2
from pathlib import Path
from collections import namedtuple


from funcnodes_opencv import OpenCVImageFormat

BASEIMAGE_BRG = cv2.imread(Path(__file__).parent / "astronaut.jpg", cv2.IMREAD_COLOR)

BASEIMAGE_BLURRED = cv2.GaussianBlur(
    BASEIMAGE_BRG,
    (21, 21),
    5,
).astype(np.uint8)

np_dtypes = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.int8,
    np.int16,
    np.int32,
    np.float16,
    np.float32,
    np.bool_,
]


def _transform(raw, request):
    if request.param.channels == 1:
        data = cv2.cvtColor(
            raw,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        data = raw.copy()
    img = OpenCVImageFormat(data)
    img.testdtype = request.param.dtype
    img.testchannels = request.param.channels
    img.raw = raw
    img.raw_transformed = data
    return img


@pytest.fixture
def image1(request):
    return _transform(BASEIMAGE_BRG, request)


@pytest.fixture
def image1_raw():
    return BASEIMAGE_BRG


@pytest.fixture
def image2(request):
    return _transform(BASEIMAGE_BLURRED, request)


@pytest.fixture
def image2_raw():
    return BASEIMAGE_BLURRED


@pytest.fixture
def image(request):
    return _transform(BASEIMAGE_BLURRED, request)


@pytest.fixture
def image_raw():
    return BASEIMAGE_BLURRED


@pytest.fixture
def mask():
    # Create a single channel 3x3 mask with alternating 255 and 0 values
    mask_raw = (
        np.random.rand(BASEIMAGE_BRG.shape[0] * BASEIMAGE_BRG.shape[1] * 1).reshape(
            BASEIMAGE_BRG.shape[0], BASEIMAGE_BRG.shape[1]
        )
        > 0.5
    ).astype(np.uint8) * 255

    img = OpenCVImageFormat(mask_raw)
    img.mask_raw = mask_raw

    return img


def pytest_generate_tests(metafunc):
    # Automatically parametrize 'image1' with the desired dtypes if it is a fixture in the test
    opts = []

    all_np_types = [
        np.uint8,
    ]

    Imageoption = namedtuple("Imageoption", ["dtype", "channels"])

    for t in all_np_types:
        opts.append(Imageoption(dtype=t, channels=3))
        opts.append(Imageoption(dtype=t, channels=1))

    if "image1" in metafunc.fixturenames:
        metafunc.parametrize(
            "image1",
            opts,
            indirect=True,
            ids=lambda opt: f"{opt.channels}c-{opt.dtype.__name__}",
        )

    # Similarly for 'image2'
    if "image2" in metafunc.fixturenames:
        metafunc.parametrize(
            "image2",
            opts,
            indirect=True,
            ids=lambda opt: f"{opt.channels}c-{opt.dtype.__name__}",
        )

    if "image" in metafunc.fixturenames:
        metafunc.parametrize(
            "image",
            opts,
            indirect=True,
            ids=lambda opt: f"{opt.channels}c-{opt.dtype.__name__}",
        )

    if "np_dtype" in metafunc.fixturenames:
        metafunc.parametrize(
            "np_dtype",
            np_dtypes,
            ids=lambda opt: f"{opt.__name__}",
        )
