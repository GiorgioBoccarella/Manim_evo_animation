from manim import *
import networkx as nx
import manimnx.manimnx as mnx
import numpy as np

# Using manimnx library

COLORS = [RED, YELLOW, GREEN, WHITE, PURPLE, LIGHT_PINK, TEAL, GOLD, ORANGE]


class ModTest(Scene):

    def construct(self):
        COLORS = [RED, YELLOW, GREEN, WHITE, PURPLE, LIGHT_PINK, TEAL, GOLD, ORANGE]

        # make a random caveman graph
        G1 = nx.relaxed_caveman_graph(8, 6, 0.20, seed=10)
        # Assign the colour to group size
        k_size = 6
        
        x = 0
        col = 0
        for edge in G1.nodes.values():
            x += 1
            edge['color'] = COLORS[col]
            if x % k_size == 0:
                col += 1
                
        
        # make the manim graph
        mng = mnx.ManimGraph(G1)

        self.add(mng) # create G1
           

class RandomGraphs(Scene):

    def construct(self):
        import time
        # list of colors to choose from
        COLORS = [RED, YELLOW, GREEN, WHITE, PURPLE, LIGHT_PINK, TEAL, GOLD, ORANGE]
        np.random.seed(int(time.time()))

        # make a random graph
        G1 = nx.relaxed_caveman_graph(8, 6, 0.20, seed=10)
        # choose random colors for the nodes
        x = 0
        col = 0
        for edge in G1.nodes.values():
            x += 1
            edge['color'] = COLORS[col]
            if x % 6 == 0:
                col += 1
                
        
        # make the manim graph
        mng = mnx.ManimGraph(G1)

        self.play(*[Create(m) for m in mng])  # create G1
        self.wait(1)

        # lets get a new graph
        G2 = G1.copy()
        #mnx.assign_positions(G2)  # assign new node positions
        
        for edge in G1.edges:
                G2.remove_edge(*edge)
                
        empty = mnx.ManimGraph(G2)        
        
        self.clear()
        self.add(empty)
        self.wait(0.5)
        
        # add and remove random edges
        new_G = nx.relaxed_caveman_graph(8, 6, 0.7, seed=10)
        G2.add_edges_from(new_G.edges)
        
        for edge in G2.edges:
            if edge not in new_G.edges:
                G2.remove_edge(*edge)

        # the transform_graph function neatly moves nodes to their new
        # positions along with edges that remain. New edges are faded in and
        # removed ones are faded out. Try to replace this with a vanilla
        # Transform and notice the difference.
        mng2 = mnx.ManimGraph(G2)
        self.add_foreground_mobject(mng2) 
        
        self.wait(0.8)
        
        G3 = G2.copy()
        mnx.assign_positions(G3)
        
        G3.add_edges_from(new_G.edges)
        for edge in G2.edges:
            if edge not in new_G.edges:
                G3.remove_edge(*edge)
        
        self.clear()
                
        self.play(*mnx.transform_graph(mng2, G3))
        self.add_foreground_mobject(mng2)


        # vanilla transform mixes up all mobjects, and doesn't look as good
        self.wait(1)
        