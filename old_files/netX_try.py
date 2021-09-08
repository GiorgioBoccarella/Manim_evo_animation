from manim import *
from manim.mobject.geometry import SmallSquare
import networkx as nx
from networkx.classes.function import neighbors
from networkx.generators import small
import numpy as np
from collections import defaultdict
import random


class ModGraph(Scene):
    def construct(self):
        vertices = [1, 2, 3, 4, 5, 6, 7, 8]
        edges = [
            (1, 7),
            (1, 8),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 8),
            (3, 4),
            (6, 1),
            (6, 2),
            (6, 3),
            (7, 2),
            (7, 4),
        ]
        g = Graph(
            vertices,
            edges,
            layout="partite",
            partitions=[[0, 1, 2]],
            layout_scale=3,
            labels=True,
            vertex_config={7: {"fill_color": RED}},
            edge_config={
                (1, 7): {"stroke_color": RED},
                (2, 7): {"stroke_color": RED},
                (4, 7): {"stroke_color": RED},
            },
        )
        self.add(g)


class PartiteGraph(Scene):
    def construct(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2, 3, 4, 6, 7, 8])
        G.add_edges_from([(0, 2), (0, 3), (1, 2), (7, 1), (6, 2), (7, 4), (7, 8)])
        graph = Graph(
            list(G.nodes),
            list(G.edges),
            layout="partite",
            partitions=[[0, 1, 4, 8]],
            labels=True,
        )
        self.add(graph)


side_l = 14
width = height = side_l
nxgraph = nx.fast_gnp_random_graph(side_l * side_l, 0)
a = [[(y * side_l + x) for x in range(side_l)] for y in range(side_l)]


def change_edges_color(obj, my_tuples, color):
    if type(my_tuples) is tuple:
        obj.add_edges(*[my_tuples], edge_config={my_tuples: {"stroke_color": color}})
    else:
        for i in range(0, len(my_tuples)):
            obj.add_edges(
                *[my_tuples[i]], edge_config={my_tuples[i]: {"stroke_color": color}}
            )


def change_vertex_color(obj, my_vertex_id, color):
    if type(my_vertex_id) is int:
        obj.add_myvertices(
            *[my_vertex_id], vertex_config={my_vertex_id: {"fill_color": color}}
        )
    else:
        for i in range(0, len(my_vertex_id)):
            obj._add_myvertices(
                *[my_vertex_id[i]],
                vertex_config={my_vertex_id[i]: {"stroke_color": color}}
            )


np.random.seed(400)


class ImportNetworkx_015(Scene):
    def construct(self):
        G = Graph.from_networkx(
            nxgraph,
            layout="partite",
            partitions=a,
            labels=False,
            vertex_type=Dot,
            layout_scale=3.5,
        )
        np.random.seed(870)

        self.add(Square(side_length=7.5))

        CN = 0.15

        main_title = Text("Mutations = ", size=0.42)
        self.add(main_title)
        main_title.move_to(UL + LEFT * 5.0 + UP * 2)

        mut_text = Text("000", size=0.42, color=PURE_RED)
        self.add(mut_text)
        mut_text.next_to(main_title, RIGHT)
        cn_strg = "Connectedness = " + str(CN)

        connect_title = Text(cn_strg, size=0.4, color=BLUE_C)
        self.add(connect_title)
        connect_title.move_to(UL + LEFT * 4.499 + UP * 1.5)

        lab_title = Text("New Phenotype = ", size=0.39)
        self.add(lab_title)
        lab_title.move_to(UL + LEFT * 4.7)
        sq = SmallSquare()
        self.add(sq)
        sq.next_to(lab_title, RIGHT)

        mut_num = 0
        t = 0

        tot_nodes = range(0, side_l * side_l)
        neighbors_dict = dict([(key, []) for key in tot_nodes])

        for x in range(0, width):
            for y in range(0, width):
                t += 1
                for x_ in range(max(0, x - 1), min(height, x + 2)):
                    for y_ in range(max(0, y - 1), min(width, y + 2)):
                        if (x, y) == (x_, y_):
                            continue
                        print(t - 1)
                        print(a[x_][y_])
                        # neighbors_dict[t - 1].append(a[x_][y_])
                        print("\n")
                        r = np.random.binomial(1, CN)
                        if r > 0:
                            # col = random.choice(COLORS)
                            G.add_edges(
                                (t - 1, a[x_][y_]),
                                edge_config={
                                    (t - 1, a[x_][y_]): {"stroke_color": WHITE}
                                },
                            )
                            neighbors_dict[t - 1].append(a[x_][y_])
                print("\n")

        for i in range(0, 6):
            rand = np.random.randint(0, side_l * side_l)
            G.add_vertices(
                *[rand],
                vertex_config={rand: {"fill_color": YELLOW}},
                vertex_type=SmallSquare
            )
            G.change_layout(layout="partite", partitions=a, layout_scale=3.5)

        # print(G.edges)
        my_keys = np.array(list(G.edges.keys()))
        unique_k = np.unique(my_keys, axis=0)
        invert_k = np.flip(unique_k, 1)
        edges_keys = np.concatenate((invert_k, unique_k))

        print(edges_keys)

        d = defaultdict(list)

        for vertex, edge in edges_keys:
            d[vertex].append(edge)

        for key, value in d.items():
            d[key] = list(set(value))

        print(d)

        for mutations in range(0, 4):
            start_point = 0
            G.add_vertices(
                *[start_point], vertex_config={start_point: {"fill_color": PURE_RED}}
            )
            G.change_layout(layout="partite", partitions=a, layout_scale=3.5)
            # Plot mutation number
            mut_num += 1
            mut_succ = Text(str(mut_num), size=0.75, color=RED)
            mut_text.become(mut_succ)
            mut_text.next_to(main_title)
            if mutations == 0:
                main_vertex = start_point
                sec_vertex = random.choice(d[main_vertex])
                edge = (main_vertex, sec_vertex)
                change_edges_color(G, edge, PURE_RED)
                G.add_vertices(
                    *[sec_vertex], vertex_config={sec_vertex: {"fill_color": RED}}
                )
                G.change_layout(layout="partite", partitions=a, layout_scale=3.5)
                self.add(G)
                self.add(mut_text)
                self.add(main_title)
                self.add(sq)
                self.add(lab_title)
                self.add(connect_title)
                self.add(Square(side_length=7.5))
                self.wait(0.15)
                self.clear()
            else:
                main_vertex = sec_vertex
                sec_vertex = random.choice(d[main_vertex])
                edge = (main_vertex, sec_vertex)
                change_edges_color(G, edge, PURE_RED)
                G.add_vertices(
                    *[sec_vertex], vertex_config={sec_vertex: {"fill_color": RED}}
                )
                G.change_layout(layout="partite", partitions=a, layout_scale=3.5)
                self.add(G)
                self.add(mut_text)
                self.add(main_title)
                self.add(sq)
                self.add(lab_title)
                self.add(connect_title)
                self.add(Square(side_length=7.5))
                self.wait(0.15)
                self.clear()

        # self.wait(1)


layout_s = 1.5


class ImportNetworkx_015_small(Scene):
    def construct(self):
        G = Graph.from_networkx(
            nxgraph,
            layout="partite",
            partitions=a,
            labels=False,
            vertex_type=Dot,
            layout_scale=layout_s,
        )
        np.random.seed(870)

        self.add(Square(side_length=7.5))

        CN = 0.15
        mut_num = 0
        t = 0

        main_title = Text("Mutations = ", size=0.42)
        self.add(main_title)
        main_title.move_to(UL + LEFT * 5.0 + UP * 2)

        mut_text = Text("000", size=0.42, color=PURE_RED)
        self.add(mut_text)
        mut_text.next_to(main_title, RIGHT)
        cn_strg = "Connectedness = " + str(CN)

        connect_title = Text(cn_strg, size=0.4, color=BLUE_C)
        self.add(connect_title)
        connect_title.move_to(UL + LEFT * 4.499 + UP * 1.5)

        lab_title = Text("New Phenotype = ", size=0.39)
        self.add(lab_title)
        lab_title.move_to(UL + LEFT * 4.7)
        sq = SmallSquare()
        self.add(sq)

        sq2 = SmallSquare()
        sq3 = SmallSquare()
        sq4 = SmallSquare()

        sq2.move_to(UP * 2 + LEFT)
        sq3.move_to(UP * 2.5 + RIGHT * 2.3)
        sq4.move_to(DOWN * 3 + RIGHT * 3)

        sq.next_to(lab_title, RIGHT)

        mut_num = 0
        t = 0

        tot_nodes = range(0, side_l * side_l)
        neighbors_dict = dict([(key, []) for key in tot_nodes])

        for x in range(0, width):
            for y in range(0, width):
                t += 1
                for x_ in range(max(0, x - 1), min(height, x + 2)):
                    for y_ in range(max(0, y - 1), min(width, y + 2)):
                        if (x, y) == (x_, y_):
                            continue
                        print(t - 1)
                        print(a[x_][y_])
                        # neighbors_dict[t - 1].append(a[x_][y_])
                        print("\n")
                        r = np.random.binomial(1, CN)
                        if r > 0:
                            # col = random.choice(COLORS)
                            G.add_edges(
                                (t - 1, a[x_][y_]),
                                edge_config={
                                    (t - 1, a[x_][y_]): {"stroke_color": WHITE}
                                },
                            )
                            neighbors_dict[t - 1].append(a[x_][y_])
                print("\n")

        for i in range(0, 6):
            rand = np.random.randint(0, side_l * side_l)
            G.add_vertices(
                *[rand],
                vertex_config={rand: {"fill_color": YELLOW}},
                vertex_type=SmallSquare
            )
            G.change_layout(layout="partite", partitions=a, layout_scale=layout_s)

        # print(G.edges)
        my_keys = np.array(list(G.edges.keys()))
        unique_k = np.unique(my_keys, axis=0)
        invert_k = np.flip(unique_k, 1)
        edges_keys = np.concatenate((invert_k, unique_k))

        print(edges_keys)

        d = defaultdict(list)

        for vertex, edge in edges_keys:
            d[vertex].append(edge)

        for key, value in d.items():
            d[key] = list(set(value))

        print(d)

        for mutations in range(0, 10):
            start_point = 0
            G.add_vertices(
                *[start_point], vertex_config={start_point: {"fill_color": PURE_RED}}
            )
            G.change_layout(layout="partite", partitions=a, layout_scale=layout_s)
            # Plot mutation number
            mut_num += 1
            mut_succ = Text(str(mut_num), size=0.75, color=RED)
            mut_text.become(mut_succ)
            mut_text.next_to(main_title)
            if mutations == 0:
                main_vertex = start_point
                sec_vertex = random.choice(d[main_vertex])
                edge = (main_vertex, sec_vertex)
                change_edges_color(G, edge, PURE_RED)
                G.add_vertices(
                    *[sec_vertex], vertex_config={sec_vertex: {"fill_color": RED}}
                )
                G.change_layout(layout="partite", partitions=a, layout_scale=layout_s)
                self.add(G)
                self.add(mut_text)
                self.add(main_title)
                self.add(sq)
                self.add(lab_title)
                self.add(connect_title)
                self.add(Square(side_length=7.5))
                self.add(sq2, sq3, sq4)
                self.wait(0.15)
                self.clear()
            else:
                main_vertex = sec_vertex
                sec_vertex = random.choice(d[main_vertex])
                edge = (main_vertex, sec_vertex)
                change_edges_color(G, edge, PURE_RED)
                G.add_vertices(
                    *[sec_vertex], vertex_config={sec_vertex: {"fill_color": RED}}
                )
                G.change_layout(layout="partite", partitions=a, layout_scale=layout_s)
                self.add(G)
                self.add(mut_text)
                self.add(main_title)
                self.add(sq)
                self.add(lab_title)
                self.add(connect_title)
                self.add(Square(side_length=7.5))
                self.add(sq2, sq3, sq4)
                self.wait(0.15)
                self.clear()


g = nx.relaxed_caveman_graph(4, 6, 0.20, seed=10)


class MyNet(Scene):
    def construct(self):

        G = nx.Graph()

        G.add_nodes_from([0, 1, 2, 3, 4])
        G.add_edges_from([(0, 2), (0, 3), (1, 2), (4, 2)])

        graph = Graph(
            list(G.nodes),
            list(G.edges),
            layout="par",
            labels=True,
            edge_config={(0, 2): {"stroke_color": RED}, (1, 2): {"stroke_color": RED}},
            vertex_config={3: {"fill_color": RED}},
        )

        graph.add_vertices(*[5], vertex_config={5: {"fill_color": RED}}, labels=True)
        graph.add_edges(*[(5, 1)])

        self.add(graph)


class RandomMove(Scene):
    CONFIG = {
        "amplitude": 0.4,
        "jiggles_per_second": 1,
    }

    def construct(self):
        points = VGroup(*[Dot(radius=0.2) for _ in range(9)])
        points.arrange_in_grid(buff=1)
        for submob in points:
            submob.jiggling_direction = rotate_vector(
                RIGHT,
                np.random.random() * TAU * 1.5,
            )
            submob.jiggling_phase = np.random.random() * TAU * 1.5

        def update_mob(mob, dt):
            for submob in mob:
                submob.jiggling_phase += dt * self.jiggles_per_second * TAU
                submob.shift(
                    self.amplitude
                    * submob.jiggling_direction
                    * np.sin(submob.jiggling_phase)
                    * dt
                )

        points.add_updater(update_mob)
        self.add(points)
        self.wait(3)


class ImportNetworkx3(ThreeDScene):
    def construct(self):
        G = Graph.from_networkx(
            nxgraph,
            layout="partite",
            partitions=a,
            labels=False,
            vertex_type=Dot,
            layout_scale=3.5,
        )
