import functools
import io
import os

from cluster_map.architecture import (
    ComposedObject,
    Layout,
    Object,
    Padding,
    Rectangle,
    Size,
)


def to_bytes(obj: Object):
    with io.BytesIO() as output:
        obj.image.save(output, format="PNG")
        return output.getvalue()


def test_fluid_layout(image_regression):
    layout = Layout(Size(2, 3), Size(20, 60), padding=Padding(0, 0, 0, 0))

    objects = [Rectangle(name=str(i)) for i in range(2 * 3)]

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="fluid_layout", layout=layout, objects=objects))
    )


def test_mix_layout(image_regression):
    layout = Layout(Size(2, 3), Size(20, 60), padding=Padding(0, 0, 0, 0))

    objects = [Rectangle(name=str(i)) for i in range(2 * 3)]

    objects[0].size = Size(5, 15)

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="mix_layout", layout=layout, objects=objects))
    )


def test_adjust_width(image_regression):
    layout = Layout(
        Size(4, 3),
        Size(120, 100),
        padding=Padding(0.1, 0.05, 0.2, 0.15),
        valign="top",
        halign="left",
    )

    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    objects[0].size = Size(2, 8)
    objects[1].size = Size(4, 2)
    objects[4].size = Size(2, 8)
    objects[5].size = Size(2, 4)
    objects[7].size = Size(2, 4)
    objects[8].size = Size(2, 8)

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="adjust_width", layout=layout, objects=objects))
    )


def test_top_left_padding(image_regression):
    layout = Layout(
        Size(4, 3),
        Size(120, 100),
        padding=Padding(0.1, 0.05, 0.2, 0.15),
        valign="top",
        halign="left",
    )

    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    objects[0].size = Size(2, 4)
    objects[1].size = Size(4, 2)
    objects[4].size = Size(4, 2)
    objects[5].size = Size(2, 4)
    objects[7].size = Size(2, 4)

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="top_left", layout=layout, objects=objects))
    )


def test_bottom_right_padding(image_regression):
    layout = Layout(
        Size(4, 3),
        Size(120, 100),
        padding=Padding(0.1, 0.05, 0.2, 0.15),
        valign="bottom",
        halign="right",
    )

    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    objects[0].size = Size(2, 4)
    objects[1].size = Size(4, 2)
    objects[4].size = Size(4, 2)
    objects[5].size = Size(2, 4)
    objects[7].size = Size(2, 4)

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="bottom_right", layout=layout, objects=objects))
    )


def test_centered_with_padding(image_regression):
    layout = Layout(
        Size(4, 3),
        Size(120, 100),
        padding=Padding(0.1, 0.05, 0.2, 0.15),
        valign="center",
        halign="center",
    )

    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    objects[0].size = Size(2, 4)
    objects[1].size = Size(4, 2)
    objects[4].size = Size(4, 2)
    objects[5].size = Size(2, 4)
    objects[7].size = Size(2, 4)

    layout.adjust_cell_sizes(objects)

    image_regression.check(
        to_bytes(ComposedObject(name="center", layout=layout, objects=objects))
    )
