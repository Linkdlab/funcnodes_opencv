from .imageformat import (
    OpenCVImageFormat,
    ImageFormat,
    conv_colorspace,
)
import funcnodes as fn
from .utils import assert_opencvdata


class ColorCodes(fn.DataEnum):
    GRAY = "GRAY"
    BGR = "BGR"
    RGB = "RGB"
    HSV = "HSV"
    LAB = "LAB"
    YUV = "YUV"
    YCrCb = "YCrCb"
    XYZ = "XYZ"
    HLS = "HLS"
    LUV = "LUV"


@fn.NodeDecorator(
    node_id="cv2.color_convert",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_render_options={"data": {"src": "out"}},
)
def color_convert(
    img: ImageFormat,
    src: ColorCodes = ColorCodes.BGR,
    trg: ColorCodes = ColorCodes.GRAY,
) -> OpenCVImageFormat:
    src = ColorCodes.v(src)
    trg = ColorCodes.v(trg)
    img = assert_opencvdata(img)
    new_data = conv_colorspace(img, src, trg)
    return OpenCVImageFormat(new_data)


NODE_SHELF = fn.Shelf(
    name="Color Nodes",
    nodes=[
        color_convert,
    ],
    description="Nodes for converting between color modes",
    subshelves=[],
)
