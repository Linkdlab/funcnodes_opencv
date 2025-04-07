from funcnodes_opencv.utils import (
    assert_similar_opencvdata,
)
import cv2
import numpy as np
from funcnodes_opencv.imageformat import (
    OpenCVImageFormat,
)
from typing import List, Literal


def make_test_simialar(
    *raws: List[np.ndarray],
):
    max_channel: List[Literal[1, 3]] = []
    for i in range(len(raws)):
        if raws[i].ndim == 2 or raws[i].shape[2] == 1:
            max_channel.append(1)
        else:
            max_channel.append(raws[i].shape[2])
    max_channel = max(max_channel)
    new_raws = []
    for raw in raws:
        if raw.ndim == 2:
            raw = raw[:, :, np.newaxis]
        if raw.shape[2] == 2:
            raw = raw[:, :, 0]
        if raw.shape[2] >= 4:
            raw = raw[:, :, :3]

        if raw.shape[2] != max_channel:
            if raw.shape[2] < max_channel:
                raw = cv2.cvtColor(raw[:, :, 0], cv2.COLOR_GRAY2BGR)
            if raw.shape[2] > max_channel:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
        new_raws.append(raw)
    return new_raws


def prep(*images):
    images = list(images)

    raws = make_test_simialar(*[i.raw_transformed for i in images])
    for i in range(len(images)):
        images[i].raw_transformed = raws[i]
    return images


def showdat(images: List[OpenCVImageFormat], *imgs: List[np.ndarray], title=None):
    imagedata = assert_similar_opencvdata(
        *[i.raw_transformed for i in images], *[i.data for i in images]
    )
    imgs = assert_similar_opencvdata(*imgs)
    if title is None:
        title = ""
    else:
        title = " " + str(title)

    # mask = np.zeros_like(imagedata[0]).astype(bool)
    mask = np.zeros((imagedata[0].shape[0], imagedata[0].shape[1]), dtype=bool)
    # set left half to 1
    # set right half to 1
    mask[:, imagedata[0].shape[1] // 2 :] = True
    for i in range(len(images)):
        raw = imagedata[i]
        img = imagedata[i + len(images)]
        out = np.zeros_like(raw)
        out[~mask] = raw[~mask]
        out[mask] = img[mask]

        if out.dtype != np.uint8:
            out = cv2.normalize(
                out,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.uint8)

        cv2.imshow(f"raw {i}" + title, out)

    if imgs:
        out = np.zeros_like(imgs[0])
        # evently distribute the images in out
        s = out.shape[1] // len(imgs)
        _h = out.shape[0] // 4
        for i in range(len(imgs)):
            raw = imgs[i]
            out[:, i * s : (i + 1) * s] = raw[:, i * s : (i + 1) * s]
        for i in range(len(imgs) - 1):
            out[:_h, (i + 1) * s, 0] = 1
            if out.shape[2] > 1:
                out[:_h, (i + 1) * s, 1] = 0
                out[:_h, (i + 1) * s, 2] = 0
        if out.dtype != np.uint8:
            out = cv2.normalize(
                out,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.uint8)

        cv2.imshow("res" + title, out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
