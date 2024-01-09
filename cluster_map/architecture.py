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
    color: tuple[int, int, int, int] = (255, 0, 0, 255)

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

    def rotate(self, degrees):
        new_image_object = type(self)(name=self.name, image_path=self.image_path)
        new_image_object._image = self._image.rotate(degrees, expand=True)
        new_image_object._size = Size(*new_image_object._image.size)
        return new_image_object


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


class Layout:
    def __init__(
        self,
        grid: Size,
        size: Size,
        valign: Literal["top", "center", "bottom"] = "top",
        halign: Literal["left", "center", "right"] = "left",
        padding: Padding | None = None,
    ):
        self.grid = grid
        self._size = size
        self.valign = valign
        self.halign = halign

        if padding is None:
            padding = Padding(0.1, 0.1, 0.1, 0.1)
        self.padding = padding

        self.heights = np.zeros(self.grid.tuple()[::-1])
        self.widths = np.zeros(self.grid.tuple()[::-1])

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, size: Size):
        self._size = size

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


class FlexibleColumnsLayout(Layout):
    def __init__(
        self,
        grid: Size,
        nrows: tuple,
        valign: Literal["top", "center", "bottom"] = "top",
        halign: Literal["left", "center", "right"] = "left",
        padding: Padding | None = None,
    ):
        super().__init__(grid, Size(0, 0), valign, halign, padding)
        self.nrows = nrows

        self.indices = self._fill_indices()
        self.cell_heights = np.ndarray(self.grid.tuple()[::-1], dtype=object)
        self.cell_widths = np.ndarray(self.grid.tuple()[::-1], dtype=object)

    def _fill_indices(self):
        indices = np.ones(self.grid.tuple()[::-1], dtype=int) * -1

        print(self.nrows)
        ith_object = 0
        for ith_row in range(self.grid.height):
            for ith_column in range(self.grid.width):
                print(ith_row, ith_column, ith_object, self.nrows[ith_column])
                if self.nrows[ith_column] > ith_row:
                    indices[ith_row, ith_column] = ith_object
                    ith_object += 1

        assert ith_object == sum(self.nrows), f"{ith_object} != {sum(self.nrows)}"

        return indices

    @property
    def size(self):
        return Size(
            width=int(
                self.widths.max(0).sum()
                + (self.padding.left + self.padding.right) * self.grid.width
            ),
            height=int(
                self.heights.sum(0).max()
                + (self.padding.top + self.padding.bottom) * self.grid.height
            ),
        )

    def adjust_cell_sizes(self, objects: list[Object]):
        # Set same width for all columns
        heights = np.zeros_like(self.heights)
        widths = np.zeros_like(self.widths)

        for i, obj in enumerate(objects):
            grid_pos = self.get_index(i)
            obj_size = obj.size
            if obj_size is None:
                raise ValueError("Size must be set for all objects")

            heights[grid_pos.y, grid_pos.x] = obj_size.height
            widths[grid_pos.y, grid_pos.x] = obj_size.width

        self.heights = heights
        self.widths = widths

        self.cell_heights = (self.heights + self.padding.top + self.padding.bottom) * (
            self.heights > 0
        )
        self.cell_heights = (
            self.cell_heights
            / self.cell_heights.sum(0)
            * self.cell_heights.sum(0).max()
        )
        self.cell_widths = np.ones_like(self.widths) * (
            self.widths.max(0)[None, :] + self.padding.left + self.padding.right
        )

        # TODO: Compute cell size based on number of items per row and on the number of grid.
        #       Total heigth and widths should be computed based on heights and widths + padding.
        #       It will be trickier to account for larger objects in some columns

    def get_index(self, index: int) -> Position:
        mask = index == self.indices
        if mask.sum() < 1:
            raise IndexError(f"Index {index} is out of bounds:\n{self.indices}")

        y, x = np.unravel_index(mask.reshape(-1).argmax(), mask.shape)

        # TODO: Verify if we should swap x and y
        return Position(x, y)

    def get_position(self, index: int) -> Position:
        grid_position = self.get_index(index)

        print(grid_position)
        print(self.cell_heights.shape)
        print(self.cell_widths.shape)

        x = (
            self.cell_widths[grid_position.y, : grid_position.x].sum()
            if grid_position.x
            else 0
        )
        y = (
            self.cell_heights[: grid_position.y, grid_position.x].sum()
            if grid_position.y
            else 0
        )

        left = self.padding.left
        right = self.padding.right
        top = self.padding.top
        bottom = self.padding.bottom

        if self.halign == "left":
            x += left
        elif self.halign == "center":
            x += self.cell_widths[grid_position.y, grid_position.x] / 2
            x -= self.widths[grid_position.y, grid_position.x] / 2
        elif self.halign == "right":
            x += self.cell_widths[grid_position.y, grid_position.x]
            x -= right + self.widths[grid_position.y, grid_position.x]

        if self.valign == "top":
            y += top
        elif self.valign == "center":
            y += self.cell_heights[grid_position.y, grid_position.x] / 2
            y -= self.heights[grid_position.y, grid_position.x] / 2
        elif self.valign == "bottom":
            y += self.cell_heights[grid_position.y, grid_position.x]
            y -= bottom + self.heights[grid_position.y, grid_position.x]

        return Position(int(x), int(y))

    # def get_position(self, index: int) -> Position:
    #     # TODO

    #     return

    # def get_size(self, index: int) -> Size:
    #     position = self.get_index(index)

    #     return Size(
    #         int(np.round(self.widths[position.y, position.x])),
    #         int(np.round(self.heights[position.y, position.x])),
    #     )


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
            # obj.size = self.layout.get_size(i)
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


@dataclass(kw_only=True)
class BoundingBox(Object):
    object: Object
    _size: None = None
    radius: float = 0.1
    fill: tuple[int, int, int, int] = (200, 200, 200, 255)
    outline: tuple[int, int, int, int] = (0, 0, 0, 255)
    width: int = 5
    padding: Padding = field(default_factory=lambda: Padding(2, 2, 2, 2))
    background_color: tuple[int, int, int, int] = (255, 255, 255, 0)

    # Will it work if size is set during construction?

    @property
    def size(self) -> Size:
        if self.object.size is None:
            raise ValueError("Size must be set for bounded object")

        return Size(
            self.object.size.width + self.padding.left + self.padding.right,
            self.object.size.height + self.padding.top + self.padding.bottom,
        )

    @property
    def image(self):
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
                int(self.padding.left),
                int(self.padding.top),
            ),
        )

        image = Image.alpha_composite(
            background,
            image,
        )

        draw = ImageDraw.Draw(image)

        draw.text(
            (self.size.width / 2, self.padding.top),
            self.name,
            fill=(0, 0, 0, 255),
            align="center",
            anchor="mt",
        )

        return image


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
