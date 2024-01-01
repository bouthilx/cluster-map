import io
import os
from pathlib import Path

import pytest

from cluster_map.architecture import ImageObject, Object, Size

ROOT = Path(os.path.dirname(__file__))


def to_bytes(obj: Object):
    with io.BytesIO() as output:
        obj.image.save(output, format="PNG")
        return output.getvalue()


def test_image_size(image_regression):
    obj = ImageObject(name="v100-sxm", image_path=ROOT / "v100_sxm.jpg")
    assert obj.size.tuple() == (1200, 636)
    image_regression.check(to_bytes(obj))


def test_image_change_size(image_regression):
    obj = ImageObject(name="v100-sxm", image_path=ROOT / "v100_sxm.jpg")
    assert obj.size.tuple() == (1200, 636)

    obj.size = Size(600, 318)
    assert obj.size.tuple() == (600, 318)
    image_regression.check(to_bytes(obj))


def test_image_change_size_wrong_ratio():
    obj = ImageObject(name="v100-sxm", image_path=ROOT / "v100_sxm.jpg")
    assert obj.size.tuple() == (1200, 636)

    with pytest.raises(ValueError, match="Size ratio"):
        obj.size = Size(800, 318)
