import os
from pathlib import Path

import manim.utils.opengl as opengl
from manim import *
from manim.opengl import *


class SurfaceExample(Scene):
    def construct(self):
        # surface_text = Text("For 3d scenes, try using surfaces")
        # surface_text.fix_in_frame()
        # surface_text.to_edge(UP)
        # self.add(surface_text)

        def param_gauss_mod(u, v):
            x = u
            y = v
            z = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x **
                                                                    3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2)
            return (x, y, z*0.3 )
        
        surface= OpenGLSurface(uv_func=param_gauss_mod, u_range= [-4,4], v_range= [-4,4], fill_opacity=1, color=BLUE)
        
        surface.set_color_by_xyz_func("z3")
        self.add(surface)
        
        self.interactive_embed()
 
    