from manimlib import *
import numpy as np
import math


class TestColor(Scene):
    def construct(self):
        cir = Square(fill_opacity=1, stroke_width=0).scale(4)

        def color_func(point: np.ndarray):
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([point[0] - mu[0], point[1] - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return [
                np.clip(abs(point[0] - point[1]), 0, 1),
                np.clip(abs(point[0] + point[1]), 0, 1),
                z,
            ]

        def color_func2(point: np.ndarray):
            return [
                np.clip(abs(point[0] - point[1]), 0, 1),
                np.clip(abs(point[0] + point[1]), 0, 1),
                np.clip(abs(point[0] + point[1]), 0, 1),
                1,
            ]

        def color_func3(point: np.ndarray):
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([point[0] - mu[0], point[1] - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return [
                np.clip(abs(point[0] - point[1]), 0, 1),
                z,
                np.clip(abs(point[0] + point[1]), 0, 1),
                1,
            ]

        def color_func4(point: np.ndarray):
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([point[0] - mu[0], point[1] - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return [
                z,
                np.clip(abs(point[0] + point[1]), 0, 1),
                np.clip(abs(point[0] + point[1]), 0, 1),
                1,
            ]

        def color_func5(point: np.ndarray):
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([point[0] - mu[0], point[1] - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return [z, np.clip(abs(point[0] + point[1]), 0, 1), z + 2]

        cir.set_color_by_rgb_func(color_func)
        self.add(cir)


class SurfaceExample(Scene):
    def construct(self):
        # surface_text = Text("For 3d scenes, try using surfaces")
        # surface_text.fix_in_frame()
        # surface_text.to_edge(UP)
        # self.add(surface_text)
        # self.wait(0.1)

        def param_gauss_mod(u, v):
            x = u
            y = v
            z = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            return (x, y, z)

        surface = Surface(
            uv_func=param_gauss_mod,
            u_range=[-4, 4],
            v_range=[-4, 4],
            fill_opacity=1,
            color=BLUE,
        )

        surface.set_color_by_xyz_func("z*1.4")

        self.add(surface)
        self.wait(7)
