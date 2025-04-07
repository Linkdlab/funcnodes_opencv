import numpy as np
import cv2
import pytest
import pytest_funcnodes

from funcnodes_opencv.colornodes import (
    color_convert,
    ColorCodes,
)
from funcnodes_opencv.utils import (
    assert_opencvdata,
)


@pytest_funcnodes.nodetest(color_convert)
@pytest.mark.parametrize(
    "code",
    [
        ColorCodes.GRAY,
        # ColorCodes.BGR,
        ColorCodes.RGB,
        ColorCodes.HSV,
        ColorCodes.LAB,
        ColorCodes.YUV,
        ColorCodes.YCrCb,
        ColorCodes.XYZ,
        ColorCodes.HLS,
        ColorCodes.LUV,
    ],
)
async def test_color_convert(image1, code):
    src = "BGR"
    if image1.testchannels == 1:
        src = "GRAY"
    conv = [f"COLOR_{src}2{ColorCodes.v(code)}"]
    if not hasattr(cv2, conv[0]):
        conv = [f"COLOR_{src}2BGR", f"COLOR_BGR2{ColorCodes.v(code)}"]

    res = (image1.raw_transformed / 255.0).astype(np.float32)
    for c in conv:
        res = cv2.cvtColor(res, getattr(cv2, c))

    res = assert_opencvdata(res)
    fnout = await color_convert.inti_call(img=image1, src=src, trg=code)
    fnoutback = (await color_convert.inti_call(img=fnout, src=code, trg=src)).data

    if code == ColorCodes.GRAY and image1.data.shape[2] == 3:
        rev_check = cv2.cvtColor(
            image1.data,
            cv2.COLOR_BGR2GRAY,
        )
        rev_check = cv2.cvtColor(
            rev_check,
            cv2.COLOR_GRAY2BGR,
        )
    else:
        rev_check = image1.data
    # showdat(
    #     [image1],
    #     image1.data,
    #     rev_check,
    #     fnoutback,
    # )
    # allow up to 4% errors since some image transformations are not exact
    np.testing.assert_allclose(fnoutback, rev_check, rtol=1e-6, atol=4e-2)
