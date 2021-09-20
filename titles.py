from manim import *
import numpy as np
from numpy.linalg.linalg import norm



class Titles(Scene):
    def construct(self):

        text = Text('Evolution on static').scale(1.5)
        text.move_to(UP*1.8)
        text_2 = Text('fitness landscape').scale(1.5)
        text_2.move_to(UP*0.7)
        self.add(text, text_2)

        self.wait(3)

class Titles2(Scene):
    def construct(self):

        text = Text('Evolution on dynamic').scale(1.5)
        text.move_to(UP*1.8)
        text_2 = Text('fitness landscape').scale(1.5)
        text_2.move_to(UP*0.7)


        self.add(text, text_2)

        self.wait(3)

class Titles3(Scene):
    def construct(self):

        text = Text('Evolution on').scale(1.5)
        text.move_to(UP*1.8)
        text_2 = Text('frequency-dependent').scale(1.5)
        text_2.move_to(UP*0.6)
        text_3 = Text('fitness landscape').scale(1.5)
        text_3.move_to(DOWN*0.6)
        self.add(text, text_2, text_3)
        self.wait(3)