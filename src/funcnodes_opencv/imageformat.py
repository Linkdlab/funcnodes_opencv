from __future__ import annotations
from typing import Literal, Tuple
import cv2
import numpy as np
from funcnodes_images.imagecontainer import register_imageformat, ImageFormat  # noqa: F401
from funcnodes_images._numpy import NumpyImageFormat
from funcnodes_images._pillow import PillowImageFormat
from funcnodes_images.utils import calc_new_size


def conv_colorspace(data: np.ndarray, from_: str, to: str) -> np.ndarray:
    if from_ == to:
        return data
    if data.ndim < 3 or data.shape[2] < 3 and from_ != "GRAY":
        data = conv_colorspace(data, "GRAY", from_)

    if from_ == "HSV" or from_ == "HLS":
        data[:, :, 0] = data[:, :, 0] * 360
    elif from_ == "LAB":
        data[:, :, 0] = data[:, :, 0] * 100
        data[:, :, 1] = data[:, :, 1] * 255 - 127
        data[:, :, 2] = data[:, :, 2] * 255 - 127
    elif from_ == "LUV":
        data[:, :, 0] = data[:, :, 0] * 100
        data[:, :, 1] = data[:, :, 1] * (134 + 220) - 134
        data[:, :, 2] = data[:, :, 2] * (140 + 122) - 140
    elif from_ == "XYZ":
        data[:, :, 0] = data[:, :, 0] * 0.950456
        data[:, :, 2] = data[:, :, 2] * 1.088754

    conv = [f"COLOR_{from_}2{to}"]
    if not hasattr(cv2, conv[0]):
        subconv1 = f"COLOR_{from_}2BGR"
        subconv2 = f"COLOR_BGR2{to}"
        if hasattr(cv2, subconv1) and hasattr(cv2, subconv2):
            conv = [subconv1, subconv2]
        else:
            raise ValueError(f"Conversion from {from_} to {to} not supported")

    for c in conv:
        # if np.issubdtype(img.dtype, np.integer):
        #     if img.dtype != np.uint8 and img.dtype != np.uint16:
        #         img = convertTo(img, np.uint16)
        data = cv2.cvtColor(data, getattr(cv2, c))

    if to == "HSV" or to == "HLS":
        data[:, :, 0] = data[:, :, 0] / 360
    elif to == "LAB":
        data[:, :, 0] = data[:, :, 0] / 100
        data[:, :, 1] = (data[:, :, 1] + 127) / 255
        data[:, :, 2] = (data[:, :, 2] + 127) / 255
    elif to == "LUV":
        data[:, :, 0] = data[:, :, 0] / 100
        data[:, :, 1] = (data[:, :, 1] + 134) / (134 + 220)
        data[:, :, 2] = (data[:, :, 2] + 140) / (140 + 122)
    elif to == "XYZ":
        data[:, :, 0] = data[:, :, 0] / 0.950456
        data[:, :, 2] = data[:, :, 2] / 1.088754
    return data


def _assert_image_channels(
    img: np.ndarray, channel: Literal[1, 3, None] = None
) -> np.ndarray:
    """Ensure the image is an opencv image and has either 1 or 3 channels. Assumed alpha channel is dropped.
    returns the image with the specified number of channels in the format [h, w, c].
    """
    if channel is None:
        channel = 1 if img.ndim <= 2 or img.shape[2] == 1 else 3
    if img.ndim > 3 or img.ndim < 2:
        raise ValueError(f"Image has {img.ndim} dimensions, expected 2 or 3")

    if img.ndim < 3:  # [h, w]
        # No channel dimension: if target is color, add a channel axis and replicate.
        img = img[:, :, np.newaxis]
        if channel == 3:
            img = conv_colorspace(img, "GRAY", "BGR")
    else:
        nchan = img.shape[2]
        # Drop any alpha channels.
        if nchan >= 4:
            # Drop the 4th channel (alpha) to keep only RGB.
            img = img[:, :, :3]
        elif nchan == 2:
            # Drop the second channel (alpha), leaving only the first channel.
            img = img[:, :, :1]
        # If the target is 3 channels but this image is grayscale (1 channel),
        # replicate the single channel to form a 3-channel image.
        if channel == 3 and img.shape[2] == 1:
            img = conv_colorspace(img, "GRAY", "BGR")
        # If the target is 1 channel but this image is color (3 channels),
        # convert to grayscale by averaging the channels.
        elif channel == 1 and img.shape[2] == 3:
            img = conv_colorspace(img, "BGR", "GRAY")
    if img.ndim < 3:  # [h, w]
        # No channel dimension: if target is color, add a channel axis and replicate.
        img = img[:, :, np.newaxis]
    return img


_int_scaling_params = {}


def get_int_scaling_params(dtype: np.dtype) -> Tuple[float, float]:
    if dtype in _int_scaling_params:
        return _int_scaling_params[dtype]

    info = np.iinfo(dtype)
    min_val = info.min
    max_val = info.max
    scale = 1.0 / (max_val - min_val)
    offset = -min_val * scale
    _int_scaling_params[dtype] = (scale, offset)
    return scale, offset


def _scale_array(arr):
    """Scale the array to the range [0., 1.]"""
    arr = np.array(arr)
    if np.issubdtype(arr.dtype, np.integer):
        scale, offset = get_int_scaling_params(arr.dtype)
        return arr.astype(np.float32) * scale + offset
    elif issubclass(arr.dtype.type, np.floating):
        narr = arr.astype(np.float32)
        # floats are only scaled to 0-1 if they are not already in that range
        if np.nanmin(narr) >= 0 and np.nanmax(narr) <= 1:
            return narr
        else:
            return cv2.normalize(
                narr,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.float32)
    elif np.issubdtype(arr.dtype, np.complexfloating):
        return _scale_array(arr.real)
    elif np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.float32)

    else:
        raise ValueError(f"Unsupported dtype: {arr.dtype}")


def _assert_opencvdata(data: np.ndarray) -> np.ndarray:
    data = _scale_array(data)
    data = _assert_image_channels(data)
    return data


class OpenCVImageFormat(NumpyImageFormat):
    def __init__(self, arr):
        # The OpenCV image format stores all images as float32 in the range [0, 1].

        super().__init__(_assert_opencvdata(arr))

    def to_jpeg(self, quality=0.75) -> bytes:
        return cv2.imencode(
            ".jpg",
            (np.clip(self.data, 0, 1) * 255).astype(np.uint8),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality * 100)],
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
    # #data = np_img.to_uint16_or_uint8()
    # if data.shape[2] == 3:
    #     cv2_img = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    # elif data.shape[2] == 4:
    #     cv2_img = cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA)
    return OpenCVImageFormat(np_img.data)


OpenCVImageFormat.add_to_converter(NumpyImageFormat, cv2_to_np)
NumpyImageFormat.add_to_converter(OpenCVImageFormat, np_to_cv2)


def cv2_to_pil(cv2_img: OpenCVImageFormat) -> PillowImageFormat:
    data = _assert_image_channels(cv2_img.data, 3)

    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return PillowImageFormat(data)


def pil_to_cv2(pil_img: PillowImageFormat) -> OpenCVImageFormat:
    np_image: NumpyImageFormat = pil_img.to_np()
    # pillow is RGB or grey
    # opencv is BGR or grey
    np_data = np_image.data

    if np_data.ndim >= 3 and np_data.shape[2] >= 3:
        np_data = np_data[:, :, :3]  # drop alpha channel if present
        np_data = cv2.cvtColor(np_data, cv2.COLOR_RGB2BGR)
    return OpenCVImageFormat(np_data)


OpenCVImageFormat.add_to_converter(PillowImageFormat, cv2_to_pil)
PillowImageFormat.add_to_converter(OpenCVImageFormat, pil_to_cv2)
