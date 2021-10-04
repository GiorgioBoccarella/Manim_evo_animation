from manim import *
import networkx as nx
import manimnx.manimnx as mnx
import numpy as np
import networkx.algorithms.community as nx_comm


np.random.seed(10)

l, k = 5, 6


def split(a, n):
    k, m = divmod(len(a), n)
    return (set(a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]) for i in range(n))


start_k = list(split(range(l * k), l))


class RandomGraphs(Scene):
    def construct(self):

        main_title = Text("Modularity Score", size=0.50)
        # self.add(main_title)
        main_title.move_to(LEFT * 4.3 + UP * 3.5)

        mod_t = DecimalNumber(0.011, num_decimal_places=3)
        # self.add(mod_t)
        mod_t.next_to(main_title, DOWN)

        # list of colors to choose from
        COLORS = [RED, YELLOW, GREEN, WHITE, PURPLE, LIGHT_PINK, TEAL, GOLD, ORANGE]

        # make a random graph
        G1 = nx.relaxed_caveman_graph(5, 6, 0, seed=10)
        my_mod = nx_comm.modularity(G1, start_k)

        mod = DecimalNumber(my_mod, num_decimal_places=3)
        mod_t.become(mod)
        mod_t.next_to(main_title, DOWN)

        x = 0
        col = 0
        for edge in G1.nodes.values():
            x += 1
            edge["color"] = COLORS[col]
            if x % 6 == 0:
                col += 1

        # make the manim graph
        mng = mnx.ManimGraph(G1)

        self.play(*[Create(m) for m in mng])  # create G1
        self.wait(1)
        c = 0
        G3 = G1.copy()
        for i in range(0, 20):
            c += 0.05
            # lets get a new graph
            G2 = G3.copy()
            # mnx.assign_positions(G2)  # assign new node positions

            for edge in G3.edges:
                G2.remove_edge(*edge)

            empty = mnx.ManimGraph(G2)

            self.clear()
            self.add(empty)

            # add and remove random edges
            new_G = nx.relaxed_caveman_graph(5, 6, c, seed=10)
            G2.add_edges_from(new_G.edges)

            for edge in G2.edges:
                if edge not in new_G.edges:
                    G2.remove_edge(*edge)

            # the transform_graph function neatly moves nodes to their new
            # positions along with edges that remain. New edges are    faded in and
            # removed ones are faded out. Try to replace this with a vanilla
            # Transform and notice the difference.

            mng2 = mnx.ManimGraph(G2)
            # self.add_foreground_mobject(mng2)

            G3 = G2.copy()
            mnx.assign_positions(G3)

            G3.add_edges_from(new_G.edges)
            for edge in G2.edges:
                if edge not in new_G.edges:
                    G3.remove_edge(*edge)

            self.clear()

            self.add(main_title)
            main_title.move_to(LEFT * 4.3 + UP * 3.5)
            mod_t = DecimalNumber(0.011, num_decimal_places=3)
            self.add(mod_t)
            mod_t.next_to(main_title, DOWN)

            my_mod = round(nx_comm.modularity(G3, start_k), 3)
            print(my_mod)
            mod = DecimalNumber(my_mod, num_decimal_places=3)
            mod_t.become(mod)
            mod_t.next_to(main_title, DOWN)

            self.play(*mnx.transform_graph(mng2, G3))
            self.add_foreground_mobject(mng2)

            self.wait(2)


class Dec(Scene):
    def construct(self):

        main_title = Text("Newman Modularity Score = ", size=0.50)
        self.add(main_title)
        main_title.move_to(RIGHT * 3 + UP * 3.5)

        mod_t = DecimalNumber(0.91, num_decimal_places=2)
        self.add(mod_t)
        mod_t.next_to(main_title, RIGHT)
