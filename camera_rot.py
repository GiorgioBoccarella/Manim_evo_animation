from manim import *
import numpy as np


class Plot(ThreeDScene):
    def construct(self):

        surf_res = 4

        surface = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            resolution=10,
            u_min=-surf_res,
            u_max=surf_res,
            v_min=-surf_res,
            v_max=surf_res,
        )

        axes = ThreeDAxes(
            x_length=(6),
            y_length=(6),
            z_length=(1.3),
            x_range=(0, 6, 1),
            y_range=(0, 6, 1),
            z_range=(0, 1, 1),
            tips=True,
        )

        # self.move_camera(0.7*np.pi/2, 0.4 * np.pi)
        self.add(axes)
        self.add(surface)

        self.begin_ambient_camera_rotation(rate=0.05, about="theta")

        for i in range(0, 20):
            self.begin_ambient_camera_rotation(rate=0.05, about="theta")
            self.wait(0.8)
            self.stop_ambient_camera_rotation()

        self.stop_ambient_camera_rotation()
