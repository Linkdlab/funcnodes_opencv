import cv2
import numpy as np
from funcnodes_images.imagecontainer import register_imageformat, ImageFormat  # noqa: F401
from funcnodes_images._numpy import NumpyImageFormat
from funcnodes_images._pillow import PillowImageFormat
from funcnodes_images.utils import calc_new_size


def _conv_colorspace(data: np.ndarray, from_: str, to: str) -> np.ndarray:
    if from_ == to:
        return data
    conv = f"COLOR_{from_}2{to}"
    if not hasattr(cv2, conv):
        raise ValueError(f"Conversion from {from_} to {to} not supported")
    return cv2.cvtColor(data, getattr(cv2, conv))


class OpenCVImageFormat(NumpyImageFormat):
    def __init__(self, arr: np.ndarray, colorspace: str = "BGR"):
        if not isinstance(arr, np.ndarray):
            raise TypeError("arr must be a numpy array")

        if arr.ndim == 2:
            colorspace = "GRAY"
        if arr.ndim == 4:  # drop alpha channel
            arr = arr[..., :3]

        if colorspace != "BGR":
            arr = _conv_colorspace(arr, colorspace, "BGR")

        super().__init__(arr)

    def to_colorspace(self, colorspace: str) -> np.ndarray:
        return _conv_colorspace(self.data, "BGR", colorspace)

    def to_jpeg(self, quality=0.75) -> bytes:
        return cv2.imencode(
            ".jpg", self.data, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)]
        )[1].tobytes()

    def to_thumbnail(self, size: tuple) -> "OpenCVImageFormat":
        return self.resize(*size, keep_ratio=True)

    def resize(
        self,
        w: int = None,
        h: int = None,
        keep_ratio: bool = True,
    ) -> "OpenCVImageFormat":  #
        new_x, new_y = calc_new_size(
            self.width(), self.height(), w, h, keep_ratio=keep_ratio
        )
        return OpenCVImageFormat(
            cv2.resize(
                self.data,
                (new_x, new_y),
            )
        )


register_imageformat(OpenCVImageFormat, "cv2")


def cv2_to_np(cv2_img: OpenCVImageFormat) -> NumpyImageFormat:
    return NumpyImageFormat(cv2_img.data)


def np_to_cv2(np_img: NumpyImageFormat) -> OpenCVImageFormat:
    return OpenCVImageFormat(np_img.to_uint8(), colorspace="RGB")


OpenCVImageFormat.add_to_converter(NumpyImageFormat, cv2_to_np)
NumpyImageFormat.add_to_converter(OpenCVImageFormat, np_to_cv2)


def cv2_to_pil(cv2_img: OpenCVImageFormat) -> PillowImageFormat:
    return cv2_img.to_np().to_img()


def pil_to_cv2(pil_img: PillowImageFormat) -> OpenCVImageFormat:
    return pil_img.to_np().to_cv2()


OpenCVImageFormat.add_to_converter(PillowImageFormat, cv2_to_pil)
PillowImageFormat.add_to_converter(OpenCVImageFormat, pil_to_cv2)
