import io

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


def test_layout(image_regression):
    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    objects[0].size = Size(2, 8)
    objects[1].size = Size(4, 2)
    objects[4].size = Size(2, 8)
    objects[5].size = Size(2, 4)
    objects[7].size = Size(2, 4)
    objects[8].size = Size(2, 8)

    obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(4, 3),
            Size(120, 100),
            padding=Padding(0.1, 0.05, 0.2, 0.15),
            valign="top",
            halign="left",
        ),
        objects=objects,
    )

    image_regression.check(to_bytes(obj))


def test_size():
    objects = [Rectangle(name=str(i)) for i in range(4 * 3)]

    obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(4, 3),
            Size(120, 100),
        ),
        objects=objects,
    )

    assert obj.size.tuple() == (120, 100)

    obj.size = Size(240, 200)
    assert obj.size.tuple() == (240, 200)
    assert obj.layout.size.tuple() == (240, 200)


def test_nested_fluid(image_regression):
    objects = [Rectangle(name=str(i)) for i in range(2)]

    left_obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(1, 2),
            Size(50, 100),
            padding=Padding(0.05, 0.1, 0.05, 0.1),
            valign="top",
            halign="left",
        ),
        objects=objects,
    )

    objects = [Rectangle(name=str(i)) for i in range(2)]

    right_obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(1, 2),
            Size(50, 100),
            padding=Padding(0.05, 0.1, 0.05, 0.1),
            valign="top",
            halign="left",
        ),
        objects=objects,
    )

    obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(2, 1),
            Size(100, 100),
            padding=Padding(0.1, 0.1, 0.1, 0.1),
            valign="top",
            halign="left",
        ),
        objects=[left_obj, right_obj],
    )

    image_regression.check(to_bytes(obj))


def test_nested_mix(image_regression):
    objects = [Rectangle(name=str(i)) for i in range(4)]
    objects[0].size = Size(10, 5)
    objects[1].size = Size(10, 5)
    objects[2].size = Size(10, 5)
    objects[3].size = Size(10, 5)

    left_obj = ComposedObject(
        name="composed",
        background_color=(0, 0, 0),
        layout=Layout(
            Size(1, 4),
            Size(40, 100),
            padding=Padding(0.05, 0.1, 0.05, 0.1),
            valign="top",
            halign="left",
        ),
        objects=objects,
    )

    objects = [Rectangle(name=str(i)) for i in range(2)]
    objects[0].size = Size(50, 50)
    objects[1].size = Size(50, 50)

    center_obj = ComposedObject(
        name="composed",
        background_color=(0, 0, 0),
        layout=Layout(
            Size(1, 2),
            Size(40, 100),
            padding=Padding(0.05, 0.1, 0.05, 0.1),
            valign="bottom",
            halign="right",
        ),
        objects=objects,
    )

    objects = [Rectangle(name=str(i)) for i in range(8)]
    for obj in objects:
        obj.size = Size(2, 20)

    right_obj = ComposedObject(
        name="composed",
        background_color=(0, 0, 0),
        layout=Layout(
            Size(4, 2),
            Size(20, 100),
            padding=Padding(0.05, 0.1, 0.05, 0.1),
            valign="center",
            halign="center",
        ),
        objects=objects,
    )

    obj = ComposedObject(
        name="composed",
        layout=Layout(
            Size(3, 1),
            Size(100, 100),
            padding=Padding(0.1, 0.1, 0.1, 0.1),
            valign="center",
            halign="center",
        ),
        objects=[left_obj, center_obj, right_obj],
    )

    image_regression.check(to_bytes(obj))
