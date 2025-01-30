import funcnodes as fn
import unittest
from funcnodes_opencv.image_processing import _findContours, _drawContours
import cv2
import funcnodes_opencv as fnocv
from funcnodes_opencv.imageformat import OpenCVImageFormat

# from funcnodes_files import FileUpload
# import base64

fn.config.IN_NODE_TEST = True


class TestImageProcessing(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.image = cv2.resize(cv2.imread("tests/astronaut.jpg"), None, fx=0.5, fy=0.5)
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

    async def test_findcontours(self):
        cnts: fn.Node = _findContours()
        cnts.inputs["img"].value = self.img
        await cnts
        cont = cnts.outputs["contours"].value
        self.assertIsInstance(cont, list)
        self.assertEqual(len(cont), 1)

    async def test_drawcontours(self):
        cnts: fn.Node = _findContours()
        cnts.inputs["img"].value = self.img
        draw_cnts: fn.Node = _drawContours()
        draw_cnts.inputs["img"].value = self.img
        draw_cnts.inputs["contours"].connect(cnts.outputs["contours"])

        await fn.run_until_complete(draw_cnts, cnts)
        image = draw_cnts.outputs["out"].value
        self.assertIsInstance(image, OpenCVImageFormat)
