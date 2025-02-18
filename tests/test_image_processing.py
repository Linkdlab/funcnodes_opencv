import funcnodes as fn
import unittest
from funcnodes_opencv.image_processing import (
    _findContours,
    _drawContours,
    _circle,
    _ellipse,
)
import cv2
import funcnodes_opencv as fnocv
from funcnodes_opencv.imageformat import OpenCVImageFormat

# from funcnodes_files import FileUpload
# import base64


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
        image_w_cnts = draw_cnts.outputs["out"].value
        self.assertIsInstance(image_w_cnts, OpenCVImageFormat)

    async def test_draw_circles(self):
        draw_circle: fn.Node = _circle()
        draw_circle.inputs["img"].value = self.img
        draw_circle.inputs["center_x"].value = [480, 215]
        draw_circle.inputs["center_y"].value = [630, 545]
        draw_circle.inputs["radius"].value = [120, 45]
        await draw_circle
        image_w_circle = draw_circle.outputs["out"].value
        self.assertIsInstance(image_w_circle, OpenCVImageFormat)

    async def test_drawellipse(self):
        draw_ellipse: fn.Node = _ellipse()
        draw_ellipse.inputs["img"].value = self.img
        draw_ellipse.inputs["center_x"].value = [480, 215]
        draw_ellipse.inputs["center_y"].value = [630, 545]
        draw_ellipse.inputs["axes_x"].value = [120, 45]
        draw_ellipse.inputs["axes_y"].value = [60, 30]
        draw_ellipse.inputs["angle"].value = [0, 45]
        await draw_ellipse
        image_w_ellipse = draw_ellipse.outputs["out"].value
        self.assertIsInstance(image_w_ellipse, OpenCVImageFormat)
