import numpy as np
from typing import Literal, List
from .imageformat import (
    OpenCVImageFormat,
    NumpyImageFormat,
    _assert_image_channels,
)
from funcnodes_images import ImageFormat


def assert_opencvimg(img) -> OpenCVImageFormat:
    if isinstance(img, OpenCVImageFormat):
        nimg = img
    elif isinstance(img, ImageFormat):
        nimg = img.to_cv2()

    elif isinstance(img, np.ndarray):
        nimg = OpenCVImageFormat(img)
    else:
        try:
            nimg = NumpyImageFormat(img).to_cv2()
        except Exception:
            raise ValueError(
                f"Image is not a valid OpenCV image. Got {type(img)} instead."
            )
    return nimg


def assert_opencvdata(img, channel: Literal[1, 3, None] = None) -> np.ndarray:
    img = assert_opencvimg(img)
    data = img.data

    data = _assert_image_channels(data, channel=channel)
    return data


def assert_similar_opencvdata(*arr) -> List[np.ndarray]:
    if len(arr) == 0:
        return arr
    arr = list(arr)
    arr = [assert_opencvdata(a) for a in arr]
    arr = match_channels(*arr)
    return tuple(arr)


def match_channels(*arr):
    """
    Standardizes the channel dimension of multiple numpy arrays so that they all have the same
    number of channels. The function assumes that the first two dimensions of each array represent
    the spatial dimensions (e.g. height and width), and the optional third dimension represents channels.

    The following rules are applied:
      - All arrays must have matching sizes for the first two dimensions.
      - Any alpha channel is dropped:
          * A 4-channel image (assumed to be RGB + alpha) is reduced to 3 channels (RGB only).
          * A 2-channel image (assumed to be grayscale with alpha) is reduced to 1 channel (grayscale).
      - After dropping alpha channels, if any image has 3 channels (color) while another has 1 channel
        (grayscale), the grayscale images are converted to 3 channels by replicating the single channel.
      - If an array has no channel dimension (ndim < 3), it is assumed to be single-channel.

    Parameters
    ----------
    *arr : numpy.ndarray
        One or more numpy arrays to standardize. Each array must have at least 2 dimensions. If a third
        dimension is present, it is interpreted as the channel dimension.

    Returns
    -------
    list
        A list of numpy arrays where each array has been adjusted so that their channel counts match.

    Raises
    ------
    ValueError
        If any of the arrays do not have matching sizes for the first two dimensions.
    """
    if len(arr) == 0:
        return arr

    arr = list(arr)

    # Verify that all arrays share the same spatial dimensions (first two dimensions).
    base_shape = arr[0].shape
    for a in arr[1:]:
        if a.shape[:2] != base_shape[:2]:
            raise ValueError(
                f"Shapes {base_shape} and {a.shape} are not matchable in the first two dimensions"
            )

    # Determine the effective channel count for each array after dropping alpha channels.
    # Rules:
    #   - If no channel dimension (ndim < 3): assume 1 channel.
    #   - 4 channels: drop the alpha (resulting in 3 channels).
    #   - 2 channels: drop the alpha (resulting in 1 channel).
    #   - 1 or 3 channels: keep as is.
    effective_channels = []
    for a in arr:
        if a.ndim < 3:
            effective_channels.append(1)
        else:
            nchan = a.shape[2]
            if nchan == 4:
                effective_channels.append(3)
            elif nchan == 2:
                effective_channels.append(1)
            else:
                effective_channels.append(nchan)

    # Determine target channel count.
    # If any image (after dropping alpha) has 3 channels, we'll standardize to 3 channels;
    # otherwise, we'll standardize to 1 channel.
    target_channels = 3 if max(effective_channels) >= 3 else 1

    # Process each array to adjust its channel count.
    for i, a in enumerate(arr):
        arr[i] = _assert_image_channels(a, channel=target_channels)

    return tuple(arr)
