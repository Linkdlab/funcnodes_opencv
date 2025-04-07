import numpy as np
import cv2
import pytest
import pytest_funcnodes
from funcnodes_images import NumpyImageFormat
import pandas as pd

from funcnodes_opencv.segmentation import (
    ContourApproximationModes,
    RetrievalModes,
    findContours,
    DistanceTypes,
    distance_transform,
    watershed,
    ConnectedComponentsAlgorithmsTypes,
    connectedComponents,
)
from funcnodes_opencv.image_processing.thresholding import auto_threshold
from funcnodes_opencv.image_processing.morphological_operations import erode


@pytest_funcnodes.nodetest(findContours)
@pytest.mark.parametrize(
    "mode",
    list(RetrievalModes),
)
@pytest.mark.parametrize(
    "method",
    list(ContourApproximationModes),
)
async def test_findContours(image1, method, mode):
    th_image = (await auto_threshold.inti_call(img=image1))[0]
    img = (th_image.data * 255).astype(np.uint8)
    th_image.raw_transformed = img

    if mode == RetrievalModes.FLOODFILL:
        cimg = img.astype(np.int32)  # cv2.FLOODFILL 32bit signed int
    else:
        cimg = img
    contours, hierarchy = cv2.findContours(
        image=cimg,
        mode=mode.value,
        method=method.value,
    )

    fout = await findContours.inti_call(
        img=th_image,
        mode=mode,
        method=method,
    )

    fout_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(fout_img, fout, -1, (0, 255, 0), 3)
    res_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(res_img, contours, -1, (0, 255, 0), 3)

    # showdat(
    #     [image1, th_image],
    #     fout_img,
    #     res_img,
    #     title=f"mode={mode}, method={method}",
    # )

    assert len(fout) == len(contours)
    for i in range(len(fout)):
        np.testing.assert_array_equal(
            fout[i],
            contours[i],
        )


@pytest_funcnodes.nodetest(distance_transform)
@pytest.mark.parametrize(
    "distance_type",
    list(DistanceTypes),
)
@pytest.mark.parametrize(
    "mask_size",
    [0, 3, 5],
)
async def test_distance_transform(image1, distance_type, mask_size):
    th_image = (await auto_threshold.inti_call(img=image1))[0]
    img = (th_image.data * 255).astype(np.uint8)
    th_image.raw_transformed = img

    if (
        distance_type not in [DistanceTypes.L2, DistanceTypes.L1, DistanceTypes.C]
        and mask_size > 0
    ):
        with pytest.raises(Exception):
            cv2.distanceTransform(img, distance_type.value, mask_size)
        return
    res = cv2.distanceTransform(img, distance_type.value, mask_size)[:, :, np.newaxis]

    fnout = (
        await distance_transform.inti_call(
            img=th_image,
            distance_type=distance_type,
            mask_size=mask_size,
        )
    ).data

    # showdat(
    #     [image1, th_image],
    #     res,
    #     fnout,
    #     title=f"distance_type={distance_type}, mask_size={mask_size}",
    # )

    np.testing.assert_allclose(
        fnout,
        res,
        rtol=1e-6,
        atol=1e-3,
    )


@pytest_funcnodes.nodetest(watershed)
async def test_watershed(image1):
    th_image = (await auto_threshold.inti_call(img=image1))[0]
    th_image = await erode.inti_call(
        img=th_image, kernel=np.ones((3, 3), np.uint8), iterations=1
    )
    img = (th_image.data * 255).astype(np.uint8)
    th_image.raw_transformed = img

    fnretval, fnlabels, fnstats = await connectedComponents.inti_call(
        img=th_image, background=1
    )

    res = cv2.watershed(
        image1.raw_transformed
        if image1.testchannels == 3
        else cv2.cvtColor(
            image1.raw_transformed,
            cv2.COLOR_GRAY2BGR,
        ),
        fnlabels.data,
    )[:, :, 0]

    fnout = await watershed.inti_call(
        img=image1,
        markers=fnlabels,
    )

    img1 = (
        image1.raw_transformed
        if image1.testchannels == 3
        else cv2.cvtColor(
            image1.raw_transformed,
            cv2.COLOR_GRAY2BGR,
        )
    ).copy()
    img2 = img1.copy()
    img1[res == -1] = [0, 0, 255]
    img2[fnout == -1] = [0, 0, 255]

    # labels = (await labels_to_color.inti_call(labels=fnout, mix=True)).data
    # showdat(
    #     [image1, th_image],
    #     # img1,
    #     # img2,
    #     labels,
    #     title="watershed",
    # )

    np.testing.assert_array_equal(
        fnout.data,
        res,
    )


@pytest_funcnodes.nodetest(connectedComponents)
@pytest.mark.parametrize(
    "connectivity",
    [4, 8],
)
@pytest.mark.parametrize("algo", list(ConnectedComponentsAlgorithmsTypes))
async def test_connectedComponents(image1, connectivity, algo):
    th_image = (await auto_threshold.inti_call(img=image1))[0]
    img = (th_image.data * 255).astype(np.uint8)
    th_image.raw_transformed = img

    fnretval, fnlabels, fnstats = await connectedComponents.inti_call(
        img=th_image,
        connectivity=connectivity,
        algorithm=algo,
    )

    assert isinstance(fnretval, int)
    assert isinstance(fnlabels, NumpyImageFormat)
    assert isinstance(fnstats, pd.DataFrame)

    (
        resretval,
        reslabels,
        _,
        _,
    ) = cv2.connectedComponentsWithStatsWithAlgorithm(
        img,
        connectivity=connectivity,
        ltype=cv2.CV_32S,
        ccltype=algo.value,
    )

    assert fnretval == resretval
    np.testing.assert_array_equal(reslabels[:, :, np.newaxis], fnlabels.data)
