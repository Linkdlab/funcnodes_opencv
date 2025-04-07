import numpy as np
import cv2
import pytest_funcnodes
from funcnodes_opencv.drawing import (
    line,
    rectangle,
    circle,
    ellipse,
    putText,
    arrowedLine,
    polylines,
    fillPoly,
    fillConvexPoly,
    drawContours,
    drawMarker,
    MarkerTypes,
    labels_to_color,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(line)
async def test_line(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.line(
        img,
        (0, 0),
        (int(image1.width() * 0.75), int(image1.height() / 2)),
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await line.inti_call(
            img=image1,
            start_x=0,
            start_y=0,
            end_x=image1.width() * 0.75,
            end_y=image1.height() / 2,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(rectangle)
async def test_rectangle(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.rectangle(
        img,
        (0, 0),
        (int(image1.width() * 0.75), int(image1.height() / 2)),
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await rectangle.inti_call(
            img=image1,
            x=0,
            y=0,
            width=image1.width() * 0.75,
            height=image1.height() / 2,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(circle)
async def test_circle(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.circle(
        img,
        (int(image1.width() / 2), int(image1.height() / 2)),
        int(image1.width() * 0.25),
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await circle.inti_call(
            img=image1,
            center_x=image1.width() / 2,
            center_y=image1.height() / 2,
            radius=image1.width() * 0.25,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(ellipse)
async def test_ellipse(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.ellipse(
        img,
        (int(image1.width() / 2), int(image1.height() / 2)),
        (40, 20),
        0,
        0,
        360,
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await ellipse.inti_call(
            img=image1,
            center_x=image1.width() / 2,
            center_y=image1.height() / 2,
            axes_x=40,
            axes_y=20,
            angle=0,
            start_angle=0,
            end_angle=360,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(putText)
async def test_putText(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.putText(
        img,
        "Hello",
        (int(image1.width() / 2 - 50), int(image1.height() / 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
        cv2.LINE_8,
        False,
    )
    res = assert_opencvdata(res)
    fnout = (
        await putText.inti_call(
            img=image1,
            text="Hello",
            org_x=image1.width() / 2 - 50,
            org_y=image1.height() / 2,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color="#FF0000",
            thickness=2,
        )
    ).data
    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(arrowedLine)
async def test_arrowedLine(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.arrowedLine(
        img,
        (0, 0),
        (int(image1.width() * 0.75), int(image1.height() / 2)),
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await arrowedLine.inti_call(
            img=image1,
            start_x=0,
            start_y=0,
            end_x=image1.width() * 0.75,
            end_y=image1.height() / 2,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(polylines)
async def test_polylines(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.polylines(
        img,
        [np.array([[10, 10], [image1.width(), 20], [30, 30]])],
        True,
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await polylines.inti_call(
            img=image1,
            pts=np.array([[10, 10], [image1.width(), 20], [30, 30]]),
            isClosed=True,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(fillPoly)
async def test_fillPoly(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.fillPoly(
        img,
        [np.array([[10, 10], [image1.width(), 20], [30, 30]])],
        (0, 0, 255),
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await fillPoly.inti_call(
            img=image1,
            pts=np.array([[10, 10], [image1.width(), 20], [30, 30]]),
            color="#FF0000",
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(fillConvexPoly)
async def test_fillConvexPoly(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.fillConvexPoly(
        img,
        np.array([[10, 10], [image1.width(), 20], [30, 30]]),
        (0, 0, 255),
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await fillConvexPoly.inti_call(
            img=image1,
            pts=np.array([[10, 10], [image1.width(), 20], [30, 30]]),
            color="#FF0000",
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(drawContours)
async def test_drawContours(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.polylines(
        img,
        [np.array([[10, 10], [image1.width(), 20], [30, 30]])],
        True,
        (0, 0, 255),
        2,
        cv2.LINE_8,
    )

    res = assert_opencvdata(res)
    fnout = (
        await polylines.inti_call(
            img=image1,
            pts=np.array([[10, 10], [image1.width(), 20], [30, 30]]),
            isClosed=True,
            color="#FF0000",
            thickness=2,
        )
    ).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(drawMarker)
async def test_drawMarker(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    res = cv2.drawMarker(
        img,
        (int(image1.width() / 2), int(image1.height() / 2)),
        markerType=cv2.MARKER_CROSS,
        markerSize=50,
        color=(0, 0, 255),
        thickness=2,
        line_type=cv2.LINE_8,
    )
    res = assert_opencvdata(res)

    fnout = (
        await drawMarker.inti_call(
            img=image1,
            pos_x=image1.width() / 2,
            pos_y=image1.height() / 2,
            markerType=MarkerTypes.CROSS,
            markerSize=50,
            color="#FF0000",
            thickness=2,
        )
    ).data
    # showdat([image1], res, fnout)

    np.testing.assert_allclose(fnout, res, rtol=1e-6)


@pytest_funcnodes.nodetest(labels_to_color)
async def test_labels_to_color():
    labels = np.array([[0, 1], [2, 0]])
    np.testing.assert_allclose(
        (await labels_to_color.inti_call(labels=labels, mix=False)).data,
        np.array(
            [[[128, 0, 0], [130, 255, 126]], [[0, 0, 128], [128, 0, 0]]], dtype=np.uint8
        )
        / 255.0,
    )
