import manim.utils.opengl as opengl
from manim import *
from manim.opengl import *


class SurfacePlot(Scene):
    def construct(self):
        resolution_fa = 70
        # self.set_camera_orientation(phi=45 * DEGREES, theta=-70 * DEGREES, distance= 5)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x * 0.5, y * 0.5, z * 0.8])

        gauss_plane = OpenGLSurface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4],
            color=BLUE,
        )

        axes = ThreeDAxes(
            x_length=(4),
            y_length=(4),
            z_length=(2),
            x_range=(0, 4, 1),
            y_range=(0, 4, 1),
            z_range=(0, 2, 1),
        )

        print(gauss_plane.get_x())
        print(gauss_plane.get_y())
        print(gauss_plane.get_z())

        self.add(gauss_plane, axes)

        dot = OpenGLSphere(radius=0.1)
        dot.set_color(color=RED)
        self.add(dot)

        gauss_plane.set_color_by_xyz_func("z*2")

        self.interactive_embed()
