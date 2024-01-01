import os

from architecture import (
    CPU,
    GPU,
    RAM,
    BoundingBox,
    Cluster,
    ComposedObject,
    Layout,
    Node,
    Padding,
    Position,
    Rectangle,
    Size,
)

print(Position(0, 1).tuple())


image_folder = "images/"
cluster_names = ["Mila", "Narval", "Beluga", "Cedar"]

padding = Padding(0.05, 0.05, 0.05, 0.05)

BoundingBox(
    name="bounding", object=Rectangle(name="rect"), size=Size(500, 1000)
).image.save("bounding_box.png")


gpus = ComposedObject(
    name="gpus",
    layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
    objects=[
        GPU(name=f"gpu{i}", image_path=os.path.join(image_folder, "v100.jpg"))
        for i in range(4)
    ],
)
gpus.image.save("gpus.png")
cpus = ComposedObject(
    name="cpus",
    layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
    objects=[
        CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
        for i in range(2)
    ],
)
cpus.image.save("cpus.png")

ram = ComposedObject(
    name="ram",
    layout=Layout(Size(2, 8), Size(100, 800)),
    objects=[
        RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.png"))
        for i in range(8 * 2)
    ],
)
ram.image.save("rams.png")

RAM(name="ram", image_path=os.path.join(image_folder, "ram.png")).image.save("ram.png")

BoundingBox(
    name="nodebbox",
    object=Node(
        name="node1",
        layout=Layout(Size(3, 1), Size(1200, 1000)),
        gpus=gpus,
        cpus=cpus,
        ram=ram,
    ),
    size=Size(1200, 1000),
    padding=padding,
).image.save("node.png", transparancy=0)

import sys

sys.exit(0)

#         name="ram{1}",
#         image_path=os.path.join(image_folder, "ram.jpg"),
#         quantity=16,
#     )
#     ],
# Node(
#     name=f"node{i}",
#     gpus=[
#         GPU(name="gpu{1}", image_path=os.path.join(image_folder, "v100.jpg"))
#     ],
#     cpus=[CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
#     ram=RAM(
#         name="ram{1}",
#         image_path=os.path.join(image_folder, "ram.jpg"),
#         quantity=16,
#     ),
# )


def build_cedar() -> Cluster:
    nodes = []
    for node in range(114):
        gpus = ComposedObject(
            name="cedar-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(name="gpu{1}", image_path=os.path.join(image_folder, "p100.jpg"))
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(1, 4), Size(100, 800)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(4)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )

    for node in range(32):
        gpus = ComposedObject(
            name="cedar-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(name="gpu{1}", image_path=os.path.join(image_folder, "p100.jpg"))
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(2, 4), Size(300, 800)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(4 * 2)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )

    for node in range(192):
        gpus = ComposedObject(
            name="cedar-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(name="gpu{1}", image_path=os.path.join(image_folder, "v100.jpg"))
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(1, 6), Size(100, 1000)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(6)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )

    cluster = Cluster(
        name="cedar",
        layout=Layout(grid=Size(19, 18), size=Size(10000, 10000)),
        nodes=nodes,
    )

    return cluster


build_cedar().image.save("cedar.png")
import sys

sys.exit(0)


def build_beluga() -> Cluster:
    nodes = []
    for node in range(172):
        gpus = ComposedObject(
            name="narval-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(
                    name="gpu{1}", image_path=os.path.join(image_folder, "v100_sxm.jpg")
                )
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(1, 6), Size(100, 1000)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(6)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )

    cluster = Cluster(
        name="beluga",
        layout=Layout(grid=Size(14, 13), size=Size(10000, 10000)),
        nodes=nodes,
    )

    return cluster


build_beluga().image.save("beluga.png")
import sys

sys.exit(0)


def build_narval() -> Cluster:
    nodes = []
    for node in range(159):
        gpus = ComposedObject(
            name="narval-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(
                    name="gpu{1}", image_path=os.path.join(image_folder, "a100_sxm.jpg")
                )
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(2, 8), Size(300, 1000)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(8 * 2)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )

    cluster = Cluster(
        name="narval",
        layout=Layout(grid=Size(13, 13), size=Size(10000, 10000)),
        nodes=nodes,
    )

    return cluster


build_narval().image.save("narval.png")
import sys

sys.exit(0)


clusters = []
for cluster_name in cluster_names:
    nodes = []
    for node in range(4):
        gpus = ComposedObject(
            name="node-{node}-gpus",
            layout=Layout(Size(1, 4), Size(400, 1000), padding=padding),
            objects=[
                GPU(name="gpu{1}", image_path=os.path.join(image_folder, "v100.jpg"))
                for i in range(4)
            ],
        )
        cpus = ComposedObject(
            name=f"node-{node}-cpus",
            layout=Layout(Size(1, 2), Size(600, 1000), padding=padding),
            objects=[
                CPU(name=f"cpu{i}", image_path=os.path.join(image_folder, "cpu.png"))
                for i in range(2)
            ],
        )
        ram = ComposedObject(
            name=f"ram-{node}-rams",
            layout=Layout(Size(2, 8), Size(300, 1000)),
            objects=[
                RAM(name=f"ram{i}", image_path=os.path.join(image_folder, "ram.jpg"))
                for i in range(8 * 2)
            ],
        )

        nodes.append(
            Node(
                name=f"node{node}",
                layout=Layout(Size(3, 1), Size(1200, 1000)),
                gpus=gpus,
                cpus=cpus,  # CPU(name="cpu{1}", image_path=os.path.join(image_folder, "cpu.png"))],
                ram=ram,  # RAM(name="ram{1}", image_path=os.path.join(image_folder, "ram.jpg"),quantity=16,)
            )
        )
        nodes[-1].image.save(f"node{node}.png")
    clusters.append(
        Cluster(
            name=cluster_name,
            layout=Layout(grid=Size(2, 2), size=Size(1000, 1000)),
            nodes=nodes,
        )
    )

# image_composition = ComposedObject(name="clusters", objects=clusters)
image_composition = clusters[0]
# print(image_composition.objects[0].nodes[0].gpus[0].size)
image_composition.image.save("test.png")
