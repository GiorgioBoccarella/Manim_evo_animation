import math
from manim import *
from manim.mobject.three_dimensions import MyInd

list = []

my_res = 1000
my_u_bound = 0.55
my_l_bound = -0.55

interval_1 = np.linspace(my_l_bound, -0.05, num=my_res)
interval_2 = np.linspace(-0.05, my_u_bound, num=my_res)

interval_red = np.linspace(253, 33, num=my_res, dtype=int)
interval_green = np.linspace(231, 145, num=my_res, dtype=int)
interval_blue = np.linspace(37, 140, num=my_res, dtype=int)

interval_red2 = np.linspace(33, 68, num=my_res, dtype=int)
interval_green2 = np.linspace(145, 1, num=my_res, dtype=int)
interval_blue2 = np.linspace(140, 84, num=my_res, dtype=int)

for i in range(my_res):
    list.append(
        tuple(
            (
                rgb_to_color(
                    [
                        interval_red[i] / 255,
                        interval_green[i] / 255,
                        interval_blue[i] / 255,
                    ]
                ),
                interval_1[i],
            )
        )
    )

for i in range(my_res):
    list.append(
        tuple(
            (
                rgb_to_color(
                    [
                        interval_red2[i] / 255,
                        interval_green2[i] / 255,
                        interval_blue2[i] / 255,
                    ]
                ),
                interval_2[i],
            )
        )
    )


class SurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 90
        self.set_camera_orientation(phi=40 * DEGREES, theta=22 * DEGREES, distance=12)

        def param_gauss(u, v):
            x = u
            y = v
            z = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            return np.array([x * 0.5, y * 0.5, z * 0.2])

        gauss_plane = ParametricSurface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4],
            fill_color=BLUE_A,
            fill_opacity=0.8,
        )

        axes = ThreeDAxes(
            x_length=(4),
            y_length=(4),
            z_length=(1),
            x_range=(0, 4, 1),
            y_range=(0, 4, 1),
            z_range=(0, 1, 1),
        )

        gauss_plane.set_style(fill_opacity=0.99)
        gauss_plane.set_fill_by_value(axes=axes, colors=list)

        self.play(Create(gauss_plane))
        self.play(Create(axes))

        self.add(axes)
        self.bring_to_front(*axes)

        dot = MyInd(center=[1, 1, 0.015], color=PURE_RED, radius=0.05)
        dot2 = MyInd(center=[1.5, 1.5, 0.015], color=PURE_RED, radius=0.05)
        dot3 = MyInd(center=[1.3, 1.4, 0.015], color=PURE_RED, radius=0.05)

        self.add(dot, dot2, dot3)
        # self.add(gauss_plane)

        self.begin_ambient_camera_rotation(rate=0.35)
        self.wait(10)
        self.stop_ambient_camera_rotation()


class FillByValueExample(ThreeDScene):
    def construct(self):
        resolution_fa = 42
        self.set_camera_orientation(phi=75 * DEGREES, theta=-120 * DEGREES)
        axes = ThreeDAxes(x_range=(0, 5, 1), y_range=(0, 5, 1), z_range=(-1, 1, 0.5))

        def param_surface(u, v):
            x = u
            y = v
            z = np.sin(x) * np.cos(y)
            return z

        surface_plane = ParametricSurface(
            lambda u, v: axes.c2p(u, v, param_surface(u, v)),
            resolution=(resolution_fa, resolution_fa),
            v_min=0,
            v_max=5,
            u_min=0,
            u_max=5,
        )
        surface_plane.set_style(fill_opacity=1)
        surface_plane.set_fill_by_value(
            axes=axes, colors=[(RED, -0.4), (YELLOW, 0), (GREEN, 0.4)]
        )
        self.add(axes, surface_plane)
