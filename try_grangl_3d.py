from manimlib import *

import numpy as np

# To watch one of these scenes, run the following:
# python -m manim example_scenes.py SquareToCircle
# Use -s to skip to the end and just save the final frame
# Use -w to write the animation to a file
# Use -o to write it to a file and open it once done
# Use -n <number> to skip ahead to the n'th animation of a scene.


class Example(Scene):
    def construct(self):
        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x * 0.3, y * 0.3, z * 0.2])

        surface = Surface(
            uv_func=param_gauss,
            u_range=[-4, 4],
            v_range=[-4, 4],
            fill_opacity=1,
            color=BLUE,
        )

        surface.set_color_by_xyz_func("z*40")

        self.play(ShowCreation(surface))
        self.wait()
        self.move_3dcamera(0.8 * np.pi / 2, -0.45 * np.pi)
        self.begin_ambient_camera_rotation()
        self.wait(6)
