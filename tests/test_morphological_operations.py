import numpy as np
import cv2
import pytest
import pytest_funcnodes

from funcnodes_opencv.image_processing.morphological_operations import (
    dilate,
    erode,
    morphologyEx,
    MorphologicalOperations,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(dilate)
async def test_dilate(image1):
    res = cv2.dilate(image1.raw_transformed, np.ones((5, 5), np.uint8), iterations=1)
    res = assert_opencvdata(res)
    fnout = (
        await dilate.inti_call(
            img=image1, kernel=np.ones((5, 5), np.uint8), iterations=1
        )
    ).data

    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-7)


@pytest_funcnodes.nodetest(erode)
async def test_erode(image1):
    res = cv2.erode(image1.raw_transformed, np.ones((5, 5), np.uint8), iterations=1)
    res = assert_opencvdata(res)
    fnout = (
        await erode.inti_call(
            img=image1, kernel=np.ones((5, 5), np.uint8), iterations=1
        )
    ).data

    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-7)


@pytest.mark.parametrize(
    "operation",
    [
        MorphologicalOperations.OPEN,
        MorphologicalOperations.CLOSE,
        MorphologicalOperations.GRADIENT,
        MorphologicalOperations.TOPHAT,
        MorphologicalOperations.BLACKHAT,
        MorphologicalOperations.HITMISS,
    ],
)
@pytest_funcnodes.nodetest(morphologyEx)
async def test_morphologyEx(image1, operation):
    fnout = (
        await morphologyEx.inti_call(
            img=image1,
            op=operation,
            kernel=np.ones((5, 5), np.uint8),
            iterations=1,
        )
    ).data

    res = cv2.morphologyEx(
        image1.raw_transformed
        if operation != MorphologicalOperations.HITMISS or image1.testchannels == 1
        else cv2.cvtColor(image1.raw_transformed, cv2.COLOR_RGB2GRAY),
        operation.value,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    res = assert_opencvdata(res)

    # showdat([image1], res, fnout)

    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=2e-1 if operation == MorphologicalOperations.HITMISS else 2e-7,
    )
