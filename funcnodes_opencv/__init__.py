from .imageformat import OpenCVImageFormat
from . import image_operations, masks, colormodes, filter, components, image_processing
import funcnodes as fn
import funcnodes_numpy as fnnp  # noqa: F401 # for type hinting

from .colormodes import ColorCodes, color_convert

__all__ = [
    "OpenCVImageFormat",
    "image_operations",
    "masks",
    "colormodes",
    "filter",
    "components",
    "NODE_SHELF",
    "ColorCodes",
    "color_convert",
]


__version__ = "0.2.3"


NODE_SHELF = fn.Shelf(
    name="OpenCV",
    description="OpenCV image processing nodes.",
    subshelves=[
        image_operations.NODE_SHELF,
        masks.NODE_SHELF,
        colormodes.NODE_SHELF,
        filter.NODE_SHELF,
        components.NODE_SHELF,
        image_processing.Image_Processing_NODE_SHELF,
    ],
    nodes=[],
)
