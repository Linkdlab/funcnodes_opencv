import funcnodes as fn
import unittest
import numpy as np
import funcnodes_opencv as fnocv
import cv2

SHOW = True
if SHOW:
    try:
        cv2.imshow("test", np.zeros((100, 100, 3), dtype=np.uint8))
    except cv2.error:
        SHOW = False


def show(img):
    if SHOW:
        if not isinstance(img, fnocv.imageformat.ImageFormat):
            img = fnocv.imageformat.OpenCVImageFormat(img)
        img = img.to_cv2().data
        img[img.sum(axis=2) == 0] = [0, 255, 0]
        cv2.imshow("test", img)
        cv2.waitKey(0)
