import sys
import os
import numpy as np
import funcnodes_opencv as fnocv
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from all_nodes_test_base import TestAllNodesBase  # noqa: E402
import test_image_processing  # noqa: E402

SHOW = False
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


class TestAllNodes(TestAllNodesBase):
    sub_test_classes = [test_image_processing.TestImageProcessing]

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
        self.assertEqual(out.data.shape, (self.img.height(), 50, 3))

        # with fx and fy
        node = fnocv.image_operations.resize()
        node.inputs["img"].value = self.img
        node.inputs["fh"].value = 0.5
        node.inputs["fw"].value = 0.5
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

    async def test_warp_affine(self):
        warpAffine = fnocv.image_operations.warpAffine()
        warpAffine.inputs["img"].value = self.sqr_img
        warpAffine.inputs["M"].value = np.array(
            [[1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        warpAffine.inputs["w"].value = self.sqr_image.data.shape[1]
        warpAffine.inputs["h"].value = self.sqr_image.data.shape[0]

        await warpAffine
        out = warpAffine.outputs["out"].value
        self.assertEqual(out.data.shape, self.sqr_image.shape)
        show(out)

    async def test_perspective_transform(self):
        perspectiveTransform = fnocv.image_operations.perpectiveTransform()
        perspectiveTransform.inputs["img"].value = self.sqr_img
        perspectiveTransform.inputs["M"].value = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
        )
        perspectiveTransform.inputs["w"].value = self.sqr_image.data.shape[1]
        perspectiveTransform.inputs["h"].value = self.sqr_image.data.shape[0]
        await perspectiveTransform
        out = perspectiveTransform.outputs["out"].value
        self.assertEqual(out.data.shape, self.sqr_image.shape)
        show(out)

    async def test_get_affine_transform_points(self):
        getAffineTransform_points = fnocv.image_operations.getAffineTransform_points()
        i1x1, i1y1 = 0, 0
        i1x2, i1y2 = 1, 0
        i1x3, i1y3 = 0, 1

        i2x1, i2y1 = 0, 0
        i2x2, i2y2 = 2, 0
        i2x3, i2y3 = 0, 2

        getAffineTransform_points.inputs["i1x1"].value = i1x1
        getAffineTransform_points.inputs["i1y1"].value = i1y1
        getAffineTransform_points.inputs["i1x2"].value = i1x2
        getAffineTransform_points.inputs["i1y2"].value = i1y2
        getAffineTransform_points.inputs["i1x3"].value = i1x3
        getAffineTransform_points.inputs["i1y3"].value = i1y3
        getAffineTransform_points.inputs["i2x1"].value = i2x1
        getAffineTransform_points.inputs["i2y1"].value = i2y1
        getAffineTransform_points.inputs["i2x2"].value = i2x2
        getAffineTransform_points.inputs["i2y2"].value = i2y2
        getAffineTransform_points.inputs["i2x3"].value = i2x3
        getAffineTransform_points.inputs["i2y3"].value = i2y3

        await getAffineTransform_points
        out = getAffineTransform_points.outputs["out"].value
        self.assertEqual(out.shape, (2, 3))

    async def test_get_perspective_transform_points(self):
        getPerspectiveTransform_points = (
            fnocv.image_operations.getPerspectiveTransform_points()
        )
        i1x1, i1y1 = 0, 0
        i1x2, i1y2 = 1, 0
        i1x3, i1y3 = 0, 1
        i1x4, i1y4 = 1, 1

        i2x1, i2y1 = 0, 0
        i2x2, i2y2 = 2, 0
        i2x3, i2y3 = 0, 2
        i2x4, i2y4 = 2, 2

        getPerspectiveTransform_points.inputs["i1x1"].value = i1x1
        getPerspectiveTransform_points.inputs["i1y1"].value = i1y1
        getPerspectiveTransform_points.inputs["i1x2"].value = i1x2
        getPerspectiveTransform_points.inputs["i1y2"].value = i1y2
        getPerspectiveTransform_points.inputs["i1x3"].value = i1x3
        getPerspectiveTransform_points.inputs["i1y3"].value = i1y3
        getPerspectiveTransform_points.inputs["i1x4"].value = i1x4
        getPerspectiveTransform_points.inputs["i1y4"].value = i1y4
        getPerspectiveTransform_points.inputs["i2x1"].value = i2x1
        getPerspectiveTransform_points.inputs["i2y1"].value = i2y1
        getPerspectiveTransform_points.inputs["i2x2"].value = i2x2
        getPerspectiveTransform_points.inputs["i2y2"].value = i2y2
        getPerspectiveTransform_points.inputs["i2x3"].value = i2x3
        getPerspectiveTransform_points.inputs["i2y3"].value = i2y3
        getPerspectiveTransform_points.inputs["i2x4"].value = i2x4
        getPerspectiveTransform_points.inputs["i2y4"].value = i2y4

        await getPerspectiveTransform_points
        out = getPerspectiveTransform_points.outputs["out"].value
        self.assertEqual(out.shape, (3, 3))

    async def test_get_affine_transform(self):
        getAffineTransform = fnocv.image_operations.getAffineTransform()
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        dst = np.array([[0, 0], [2, 0], [0, 2]], dtype=np.float32)
        getAffineTransform.inputs["src"].value = src
        getAffineTransform.inputs["dst"].value = dst

        await getAffineTransform
        out = getAffineTransform.outputs["out"].value
        self.assertEqual(out.shape, (2, 3))

    async def test_get_perspective_transform(self):
        getPerspectiveTransform = fnocv.image_operations.getPerspectiveTransform()
        src = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
        dst = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float32)
        getPerspectiveTransform.inputs["src"].value = src
        getPerspectiveTransform.inputs["dst"].value = dst

        await getPerspectiveTransform
        out = getPerspectiveTransform.outputs["out"].value
        self.assertEqual(out.shape, (3, 3))

    async def test_threshold(self):
        threshold = fnocv.masks.threshold()
        threshold.inputs["img"].value = self.img
        threshold.inputs["thresh"].value = 127
        threshold.inputs["maxval"].value = 255
        threshold.inputs["type"].value = fnocv.masks.ThresholdTypes.BINARY

        await threshold
        out = threshold.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_auto_threshold(self):
        auto_threshold = fnocv.masks.auto_threshold()
        auto_threshold.inputs["img"].value = self.img
        auto_threshold.inputs["maxval"].value = 255
        auto_threshold.inputs["type"].value = fnocv.masks.AutoThresholdTypes.OTSU

        await auto_threshold
        out = auto_threshold.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_inrange_single_channel(self):
        inrange = fnocv.masks.in_range_singel_channel()
        inrange.inputs["img"].value = self.img
        inrange.inputs["lower"].value = 100
        inrange.inputs["upper"].value = 200

        await inrange

        out = inrange.outputs["out"].value
        self.assertEqual(out.data.shape[:2], self.image.shape[:2])

    async def test_inrange(self):
        inrange = fnocv.masks.in_range()
        inrange.inputs["img"].value = self.img
        inrange.inputs["lower_c1"].value = 100
        inrange.inputs["upper_c1"].value = 200
        inrange.inputs["lower_c2"].value = 100
        inrange.inputs["upper_c2"].value = 200
        inrange.inputs["lower_c3"].value = 100
        inrange.inputs["upper_c3"].value = 200

        await inrange
        out = inrange.outputs["out"].value
        self.assertEqual(out.data.shape[:2], self.image.shape[:2])

    async def test_adaptive_threshold(self):
        adaptive_threshold = fnocv.masks.adaptive_threshold()
        adaptive_threshold.inputs["img"].value = self.img
        adaptive_threshold.inputs["maxval"].value = 255
        adaptive_threshold.inputs["block_size"].value = 3
        adaptive_threshold.inputs["c"].value = 2

        await adaptive_threshold
        out = adaptive_threshold.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_distance_transform(self):
        distance_transform = fnocv.masks.distance_transform()
        distance_transform.inputs["img"].value = self.img
        distance_transform.inputs["distance_type"].value = fnocv.masks.DistanceTypes.L2
        distance_transform.inputs["mask_size"].value = 3

        await distance_transform
        out = distance_transform.outputs["out"].value
        self.assertEqual(out.data.shape[:2], self.image.shape[:2])
        self.assertEqual(out.data.shape[2], 1)
        show(out)

    async def test_watershed(self):
        watershed = fnocv.masks.watershed()
        watershed.inputs["img"].value = self.img
        watershed.inputs["markers"].value = self.img
        await watershed
        out = watershed.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape[:2])
        show(out)

    async def test_color_convert(self):
        color_convert = fnocv.colormodes.color_convert()
        color_convert.inputs["img"].value = self.img
        color_convert.inputs["code"].value = fnocv.ColorCodes.GRAY
        await color_convert
        out = color_convert.outputs["out"].value
        self.assertEqual(out.data.shape[:2], self.image.shape[:2])
        self.assertEqual(out.data.shape[2], 1)
        show(out)

    async def test_filter2d(self):
        filter2d = fnocv.filter.filter2D()
        filter2d.inputs["img"].value = self.img
        filter2d.inputs["kernel"].value = np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        )
        await filter2d
        out = filter2d.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_bilateral_filter(self):
        bilateral_filter = fnocv.filter.bilateralFilter()
        bilateral_filter.inputs["img"].value = self.img
        bilateral_filter.inputs["d"].value = 9
        bilateral_filter.inputs["sigmaColor"].value = 75
        bilateral_filter.inputs["sigmaSpace"].value = 75
        await bilateral_filter
        out = bilateral_filter.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_blur(self):
        blur = fnocv.filter.blur()
        blur.inputs["img"].value = self.img
        blur.inputs["kw"].value = 5
        blur.inputs["kh"].value = 5
        await blur
        out = blur.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_box_filter(self):
        box_filter = fnocv.filter.boxFilter()
        box_filter.inputs["img"].value = self.img
        box_filter.inputs["kw"].value = 5
        box_filter.inputs["kh"].value = 5
        box_filter.inputs["normalize"].value = True
        await box_filter
        out = box_filter.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_dilate(self):
        dilate = fnocv.filter.dilate()
        dilate.inputs["img"].value = self.img
        await dilate
        out = dilate.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_erode(self):
        erode = fnocv.filter.erode()
        erode.inputs["img"].value = self.img
        await erode
        out = erode.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_gaussian_blur(self):
        gaussian_blur = fnocv.filter.GaussianBlur()
        gaussian_blur.inputs["img"].value = self.img
        gaussian_blur.inputs["kw"].value = 5
        gaussian_blur.inputs["kh"].value = 5
        gaussian_blur.inputs["sigmaX"].value = 0
        gaussian_blur.inputs["sigmaY"].value = 0
        await gaussian_blur
        out = gaussian_blur.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_laplacian(self):
        laplacian = fnocv.filter.Laplacian()
        laplacian.inputs["img"].value = self.img
        await laplacian
        out = laplacian.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_median_blur(self):
        median_blur = fnocv.filter.medianBlur()
        median_blur.inputs["img"].value = self.img
        median_blur.inputs["ksize"].value = 5
        await median_blur
        out = median_blur.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_pyr_down(self):
        pyr_down = fnocv.filter.pyrDown()
        pyr_down.inputs["img"].value = self.img
        await pyr_down
        out = pyr_down.outputs["out"].value
        self.assertEqual(
            out.data.shape, (self.image.shape[0] // 2, self.image.shape[1] // 2, 3)
        )
        show(out)

    async def test_pyr_up(self):
        pyr_up = fnocv.filter.pyrUp()
        pyr_up.inputs["img"].value = self.img
        await pyr_up
        out = pyr_up.outputs["out"].value
        self.assertEqual(
            out.data.shape, (self.image.shape[0] * 2, self.image.shape[1] * 2, 3)
        )
        show(out)

    async def test_scharr(self):
        scharr = fnocv.filter.Scharr()
        scharr.inputs["img"].value = self.img
        scharr.inputs["dx"].value = 1
        scharr.inputs["dy"].value = 0
        scharr.inputs["scale"].value = 1
        scharr.inputs["delta"].value = 0
        await scharr
        out = scharr.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_sobel(self):
        sobel = fnocv.filter.Sobel()
        sobel.inputs["img"].value = self.img
        sobel.inputs["dx"].value = 1
        sobel.inputs["dy"].value = 0
        sobel.inputs["ksize"].value = 3
        sobel.inputs["scale"].value = 1
        sobel.inputs["delta"].value = 0
        await sobel
        out = sobel.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_stack_blur(self):
        stack_blur = fnocv.filter.stackBlur()
        stack_blur.inputs["img"].value = self.img
        stack_blur.inputs["kw"].value = 5
        await stack_blur
        out = stack_blur.outputs["out"].value
        self.assertEqual(out.data.shape, self.image.shape)
        show(out)

    async def test_labels_to_color(self):
        labels_to_color = fnocv.components.labels_to_color()
        labels_to_color.inputs["labels"].value = np.array(
            [[0, 1], [1, 0]], dtype=np.int32
        )
        await labels_to_color
        out = labels_to_color.outputs["out"].value
        self.assertEqual(out.data.shape, (2, 2, 3))
        self.assertEqual(out.data.dtype, np.uint8)
