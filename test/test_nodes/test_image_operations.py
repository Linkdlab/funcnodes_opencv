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

    def test_show(self):
        show(self.img)

    async def test_resize(self):
        # with w and h
        node = fnocv.image_operations.resize()
        node.inputs["img"].value = self.img
        node.inputs["w"].value = 50
        node.inputs["h"].value = 50
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (50, 50, 3))

        # with missing h
        node = fnocv.image_operations.resize()
        node.inputs["img"].value = self.img
        node.inputs["w"].value = 50
        await node
        out = node.outputs["out"].value
        self.assertEqual(out, fn.NoValue)

        # with fx and fy
        node = fnocv.image_operations.resize()
        node.inputs["img"].value = self.img
        node.inputs["fx"].value = 0.5
        node.inputs["fy"].value = 0.5
        await node
        out = node.outputs["out"].value
        self.assertEqual(
            out.data.shape, (self.image.shape[0] // 2, self.image.shape[1] // 2, 3)
        )

        if SHOW:
            cv2.imshow("test", out.data)
            cv2.waitKey(0)

    async def test_flip(self):
        # horizontal
        node = fnocv.image_operations.flip()
        node.inputs["img"].value = self.img
        node.inputs["flip_code"].value = fnocv.image_operations.FlipCodes.HORIZONTAL
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[0], self.image.shape[1], 3))
        np.testing.assert_array_equal(out.data, self.image[:, ::-1])
        # vertical
        node = fnocv.image_operations.flip()
        node.inputs["img"].value = self.img
        node.inputs["flip_code"].value = fnocv.image_operations.FlipCodes.VERTICAL
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[0], self.image.shape[1], 3))
        np.testing.assert_array_equal(out.data, self.image[::-1])

        # both
        node = fnocv.image_operations.flip()
        node.inputs["img"].value = self.img
        node.inputs["flip_code"].value = fnocv.image_operations.FlipCodes.BOTH
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[0], self.image.shape[1], 3))
        np.testing.assert_array_equal(out.data, self.image[::-1, ::-1])

        if SHOW:
            cv2.imshow("test", out.data)
            cv2.waitKey(0)

    async def test_rotate(self):
        # 90 clockwise
        node = fnocv.image_operations.rotate()
        node.inputs["img"].value = self.img
        node.inputs[
            "rot"
        ].value = fnocv.image_operations.RoationCode.ROTATE_90_CLOCKWISE
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[1], self.image.shape[0], 3))
        np.testing.assert_array_equal(out.data, np.rot90(self.image, 3))

        # 180
        node = fnocv.image_operations.rotate()
        node.inputs["img"].value = self.img
        node.inputs["rot"].value = fnocv.image_operations.RoationCode.ROTATE_180
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[0], self.image.shape[1], 3))
        np.testing.assert_array_equal(out.data, np.rot90(self.image, 2))

        # 90 counterclockwise
        node = fnocv.image_operations.rotate()
        node.inputs["img"].value = self.img
        node.inputs[
            "rot"
        ].value = fnocv.image_operations.RoationCode.ROTATE_90_COUNTERCLOCKWISE
        await node
        out = node.outputs["out"].value
        self.assertEqual(out.data.shape, (self.image.shape[1], self.image.shape[0], 3))
        np.testing.assert_array_equal(out.data, np.rot90(self.image))

        if SHOW:
            cv2.imshow("test", out.data)
            cv2.waitKey(0)

    async def test_free_rotation(self):
        angle = 45
        freeRotation = fnocv.image_operations.freeRotation()
        freeRotation.inputs["img"].value = self.sqr_img
        freeRotation.inputs["angle"].value = angle

        # keep
        freeRotation.inputs[
            "mode"
        ].value = fnocv.image_operations.FreeRotationCropMode.KEEP
        await freeRotation
        out = freeRotation.outputs["out"].value

        diag = int(np.sqrt(self.sqr_image.shape[0] ** 2 + self.sqr_image.shape[1] ** 2))
        self.assertEqual(out.data.shape, (diag, diag, 3))
        show(out)

        # None
        freeRotation.inputs[
            "mode"
        ].value = fnocv.image_operations.FreeRotationCropMode.NONE
        await freeRotation
        out = freeRotation.outputs["out"].value
        self.assertEqual(
            out.data.shape, (self.sqr_image.shape[0], self.sqr_image.shape[1], 3)
        )
        show(out)

        # crop
        freeRotation.inputs["angle"].value = 45
        freeRotation.inputs[
            "mode"
        ].value = fnocv.image_operations.FreeRotationCropMode.CROP
        await freeRotation
        out = freeRotation.outputs["out"].value
        diag = int(
            np.sqrt(
                (self.sqr_image.shape[0] / 2) ** 2 + (self.sqr_image.shape[1] / 2) ** 2
            )
        )
        self.assertEqual(out.data.shape, (diag, diag, 3))
        show(out)

        print(freeRotation._repr_json_())
