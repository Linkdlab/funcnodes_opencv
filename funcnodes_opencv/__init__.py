from .imageformat import OpenCVImageFormat
from . import image_operations, masks, colormodes, filter, components
import funcnodes as fn
import funcnodes_numpy as fnnp  # noqa: F401 # for type hinting

__all__ = ["OpenCVImageFormat", "image_operations", "masks"]


__version__ = "0.1.6"


NODE_SHELF = fn.Shelf(
    name="OpenCV",
    description="OpenCV image processing nodes.",
    subshelves=[
        image_operations.NODE_SHELF,
        masks.NODE_SHELF,
        colormodes.NODE_SHELF,
        filter.NODE_SHELF,
        components.NODE_SHELF,
    ],
    nodes=[],
)
