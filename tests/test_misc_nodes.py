import numpy as np
import cv2
import pytest
import pytest_funcnodes
from funcnodes_opencv.colornodes import color_convert
from funcnodes_opencv.image_operations.normalization_equalization import CLAHE
from funcnodes_opencv.misc_nodes import (
    replace_channel,
    minmax_lcn,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest.mark.parametrize(
    "ksize",
    np.arange(2, 20, 5),
)
@pytest.mark.parametrize(
    "mincontrast",
    np.arange(
        0,
        1.1,
        0.25,
    ),
)
@pytest_funcnodes.nodetest(minmax_lcn)
async def test_minmax_lcn(image1, ksize, mincontrast):
    # res = cv2.minMaxLoc(image1.raw_transformed)
    # res = assert_opencvdata(res)
    fnout = (
        await minmax_lcn.inti_call(img=image1, ksize=ksize, mincontrast=mincontrast)
    ).data

    # showdat([image1], fnout, title=f"ksize={ksize}, mincontrast={mincontrast}")
    assert isinstance(fnout, np.ndarray)
    # np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-7)


@pytest_funcnodes.nodetest(replace_channel)
@pytest.mark.parametrize(
    "channel",
    [1, 2, 3],
)
async def test_replace_channel(image1, image2, channel):
    res = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    i2 = image2.raw_transformed.copy()
    if image2.testchannels == 1:
        i2 = cv2.cvtColor(i2, cv2.COLOR_GRAY2BGR)

    res[:, :, channel - 1] = i2[:, :, channel - 1]
    res = assert_opencvdata(res)
    fnout = (
        await replace_channel.inti_call(trg_img=image1, src_img=image2, channel=channel)
    ).data

    # showdat([image1, image2], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-7)


@pytest_funcnodes.nodetest(replace_channel)
@pytest.mark.parametrize(
    "channel, mode",
    [(3, "HSV"), (2, "HLS"), (1, "YCrCb"), (1, "LAB")],
)
async def test_replace_channel_clahe(image1, channel, mode):
    cconv = await color_convert.inti_call(
        img=image1,
        src="BGR",
        trg=mode,
    )

    clahe = await CLAHE.inti_call(
        img=image1,
        clip_limit=400,
        tile_grid_size=(8, 8),
    )

    repl = await replace_channel.inti_call(
        trg_img=cconv,
        src_img=clahe,
        channel=channel,
    )

    _ = await color_convert.inti_call(
        img=repl,
        src=mode,
        trg="BGR",
    )

    # showdat(
    #     [image1],
    #     image1.data,
    #     bcconv.data,
    #     clahe.data,
    # )
