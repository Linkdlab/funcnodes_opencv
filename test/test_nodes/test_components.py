import funcnodes as fn
import unittest
import numpy as np
import funcnodes_opencv as fnocv
import cv2

from . import SHOW, show


class TestImageOperations(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.image = cv2.resize(cv2.imread("test/astronaut.jpg"), None, fx=0.5, fy=0.5)
        self.image[self.image.sum(axis=2) == 0] = [1, 1, 1]
        if self.image.shape[0] % 2 != 0:
            self.image = self.image[:-1]
        if self.image.shape[1] % 2 != 0:
            self.image = self.image[:, :-1]

        self.sqr_image = self.image[
            : min(self.image.shape[0], self.image.shape[1]),
            : min(self.image.shape[0], self.image.shape[1]),
        ]
        self.img = fnocv.imageformat.OpenCVImageFormat(self.image)
        self.sqr_img = fnocv.imageformat.OpenCVImageFormat(self.sqr_image)

    async def test_connected_components(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        retval, labels = cv2.connectedComponents(img, connectivity=8)

        show(fnocv.imageformat.NumpyImageFormat(labels))
        self.assertEqual(retval, 2)
        node = fnocv.components.connectedComponents()
        node.inputs["img"].value = self.img
        await node
        out = node.outputs["labels"].value
        self.assertEqual(out.shape, self.image.shape[:2])
        self.assertEqual(out.dtype, np.int32)
        self.assertEqual(out.max(), 1)
