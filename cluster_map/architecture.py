from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class Position:
    x: int = 0
    y: int = 0

    def tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass
class Size:
    width: int
    height: int

    def tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class Padding:
    top: float
    right: float
    bottom: float
    left: float

    def tuple(self) -> tuple[float, float, float, float]:
        return (self.top, self.right, self.bottom, self.left)


@dataclass(kw_only=True)
class Object:
    name: str
    _size: Size | None = None

    @property
    def size(self) -> Size | None:
        return self._size

    @size.setter
    def size(self, size: Size | None):
        self._size = size

    @property
    def image(self) -> Image.Image:
        ...


@dataclass(kw_only=True)
class Rectangle(Object):
    color: tuple[int, int, int] = (255, 0, 0)

    @property
    def image(self) -> Image.Image:
        assert self.size is not None
        return Image.new("RGB", self.size.tuple(), color=self.color)


T = TypeVar("T", bound=Object)


image_cache = {}


def open_image(image_path: str) -> Image.Image:
    if image_path not in image_cache:
        image_cache[image_path] = Image.open(image_path)

    return image_cache[image_path]


@dataclass(kw_only=True)
class ImageObject(Object):
    image_path: str
    _image: Image.Image = field(init=False)

    def __post_init__(self):
        self._image = open_image(self.image_path)
        self._size = Size(*self._image.size)

    @property
    def image(self) -> Image.Image:
        assert self.size is not None
        return self._image.resize(self.size.tuple())

    @property
    def size(self) -> Size | None:
        return self._size

    @size.setter
    def size(self, size: Size):
        (width, height) = self._image.size
        ratio = width / height
        if not np.isclose(size.width / size.height, ratio, rtol=10e-2):
            raise ValueError(
                f"Size ratio ({size.width}, {size.height}: {size.width / size.height}) does not match image ratio ({width}, {height}: {ratio})"
            )
        self._size = size


@dataclass(kw_only=True)
class GPU(ImageObject):
    ...


@dataclass(kw_only=True)
class CPU(ImageObject):
    ...


@dataclass(kw_only=True)
class RAM(ImageObject):
    # quantity: int
    ...

    def __post_init__(self):
        self._image = open_image(self.image_path).rotate(90, expand=True)
        self._size = Size(*self._image.size)


@dataclass
class Layout:
    grid: Size
    size: Size
    valign: Literal["top", "center", "bottom"] = "top"
    halign: Literal["left", "center", "right"] = "left"
    padding: Padding = field(default_factory=lambda: Padding(0.1, 0.1, 0.1, 0.1))
    heights: np.ndarray = field(init=False)
    widths: np.ndarray = field(init=False)

    def __post_init__(self):
        self.heights = np.zeros(self.grid.tuple()[::-1])
        self.widths = np.zeros(self.grid.tuple()[::-1])

    def adjust_cell_sizes(self, objects: list[Object]):
        # Set same width for all columns
        available_height = self.size.height * (
            1 - self.padding.top - self.padding.bottom
        )
        available_width = self.size.width * (1 - self.padding.left - self.padding.right)

        heights = np.ones_like(self.heights) * available_height / self.grid.height

        widths = np.zeros_like(self.widths)
        ratios = np.zeros_like(self.widths)

        for i, obj in enumerate(objects):
            grid_pos = self.get_index(i)
            obj_size = obj.size
            if obj_size is None:
                ratio = np.nan
                width = available_width / self.grid.width
            else:
                ratio = obj_size.width / obj_size.height
                width = heights[grid_pos.y, grid_pos.x] * ratio
            ratios[grid_pos.y, grid_pos.x] = ratio
            widths[grid_pos.y, grid_pos.x] = width

        if widths.max(0).sum() > available_width:
            to_adjust = widths > available_width / self.grid.width
            fix_columns = to_adjust.sum(0) == 0
            fix_width = (widths.max(0) * fix_columns).sum()
            to_adjust_columns = to_adjust.sum(0) > 0
            to_adjust_widths = to_adjust_columns * widths.max(0)
            adjusted_widths = (
                (available_width - fix_width)
                * to_adjust_widths
                / to_adjust_widths.sum()
            )

            widths = adjusted_widths * to_adjust + widths * (1 - to_adjust)
            heights = widths / ratios

            heights = np.nan_to_num(heights, nan=0)
            row_heights = heights.max(1)
            unset_heights = heights == 0
            heights = unset_heights * row_heights[:, None] + (~unset_heights) * heights

            unset_row_heights = row_heights == 0

            if np.any(unset_row_heights):
                heights = (
                    np.ones_like(heights)
                    * (
                        unset_row_heights
                        * (
                            available_height
                            - heights.max(1).sum() / unset_row_heights.sum()
                        )
                    )[:, None]
                    + heights * (~unset_row_heights)[:, None]
                )

        assert heights.max(1).sum() <= available_height
        self.heights = heights
        self.widths = widths

    def get_index(self, index: int) -> Position:
        if index >= (self.grid.width * self.grid.height):
            raise IndexError(f"Index {index} is out of bounds: {self.grid}")
        x = index % self.grid.width
        y = index // self.grid.width

        return Position(x, y)

    def get_position(self, index: int) -> Position:
        grid_position = self.get_index(index)

        x = self.widths[:, : grid_position.x].max(0).sum() if grid_position.x else 0
        y = self.heights[: grid_position.y].max(1).sum() if grid_position.y else 0

        left = self.padding.left * self.size.width / self.grid.width
        right = self.padding.right * self.size.width / self.grid.width
        top = self.padding.top * self.size.height / self.grid.height
        bottom = self.padding.bottom * self.size.height / self.grid.height

        x += (left + right) * grid_position.x
        y += (top + bottom) * grid_position.y

        if self.halign == "left":
            x += left
        elif self.halign == "center":
            x += (left + right + self.widths[:, grid_position.x].max()) / 2
            x -= self.widths[grid_position.y, grid_position.x] / 2
        elif self.halign == "right":
            x += left + right + self.widths[:, grid_position.x].max()
            x -= right + self.widths[grid_position.y, grid_position.x]

        if self.valign == "top":
            y += top
        elif self.valign == "center":
            y += (top + bottom + self.heights[grid_position.y].max()) / 2
            y -= self.heights[grid_position.y, grid_position.x] / 2
        elif self.valign == "bottom":
            y += top + bottom + self.heights[grid_position.y].max()
            y -= bottom + self.heights[grid_position.y, grid_position.x]

        return Position(int(x), int(y))

    def get_size(self, index: int) -> Size:
        position = self.get_index(index)

        return Size(
            int(np.round(self.widths[position.y, position.x])),
            int(np.round(self.heights[position.y, position.x])),
        )


@dataclass(kw_only=True)
class ComposedObject(Object, Generic[T]):
    layout: Layout = field(default_factory=Layout)
    objects: list[T] = field(default_factory=list)
    background_color: tuple[int, int, int] = (255, 255, 255, 0)

    def __post_init__(self):
        self.layout.adjust_cell_sizes(self.objects)

    @property
    def image(self) -> Image.Image:
        # TODO: Support border with name
        # TODO: Support background color
        image = Image.new("RGBA", self.layout.size.tuple(), color=self.background_color)
        for i, obj in enumerate(self.objects):
            obj_position = self.layout.get_position(i)
            obj.size = self.layout.get_size(i)
            image.paste(obj.image, obj_position.tuple())

        return image

    @property
    def size(self) -> Size:
        return self.layout.size

    @size.setter
    def size(self, size: Size):
        self.layout.size = size
        # TODO: Test nested ComposedObject
        self.layout.adjust_cell_sizes(self.objects)


def BoundingBox(size: Size, **kwargs):
    bounding_box = _BoundingBox(**kwargs)
    bounding_box.size = size
    return bounding_box


@dataclass(kw_only=True)
class _BoundingBox(Object):
    object: Object
    _size: Size | None = None
    radius: float = 0.1
    fill: tuple[int, int, int, int] = (200, 200, 200, 255)
    outline: tuple[int, int, int, int] = (0, 0, 0, 255)
    width: int = 5
    padding: Padding = field(default_factory=lambda: Padding(0.1, 0.1, 0.1, 0.1))
    background_color: tuple[int, int, int, int] = (255, 255, 255, 0)

    # Will it work if size is set during construction?

    @property
    def size(self) -> Size | None:
        return self._size

    @size.setter
    def size(self, size: Size | None):
        self._size = size
        self.object.size = Size(
            int(size.width * (1 - self.padding.left - self.padding.right)),
            int(size.height * (1 - self.padding.top - self.padding.bottom)),
        )

    @property
    def image(self):
        assert self.size is not None
        background = Image.new("RGBA", self.size.tuple(), color=self.background_color)
        draw = ImageDraw.Draw(background)
        draw.rounded_rectangle(
            ((0, 0), self.size.tuple()),
            radius=int(self.radius * (self.size.width + self.size.height) / 2),
            fill=self.fill,
            outline=self.outline,
            width=self.width,
            corners=None,
        )

        image = Image.new("RGBA", self.size.tuple(), color=(255, 255, 255, 0))
        image.paste(
            self.object.image,
            (
                int(self.padding.left * self.size.width),
                int(self.padding.top * self.size.height),
            ),
        )

        return Image.alpha_composite(
            background,
            image,
        )


@dataclass(kw_only=True)
class Node(ComposedObject):
    gpus: ComposedObject[GPU]
    cpus: ComposedObject[CPU]
    ram: ComposedObject[RAM]

    objects: list[ComposedObject] = field(init=False)

    def __post_init__(self):
        self.objects = [self.gpus, self.cpus, self.ram]
        super().__post_init__()

    @property
    def image(self):
        image = super().image
        self.size
        background = Image.new("RGBA", self.size.tuple(), color=self.background_color)
        draw = ImageDraw.Draw(background)
        draw.rounded_rectangle(
            ((0, 0), self.size.tuple()),
            radius=90,
            fill=(200, 200, 200),
            outline=None,
            width=5,
            corners=None,
        )

        background.paste(image, (0, 0))
        return background


@dataclass(kw_only=True)
class Cluster(ComposedObject):
    nodes: list[Node]

    objects: list[Node] = field(init=False)

    def __post_init__(self):
        self.objects = self.nodes
        super().__post_init__()
