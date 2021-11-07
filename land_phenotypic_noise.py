from manim import *
from numpy.random.mtrand import rand
from rich.console import group


class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 42
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u
            y = v
            sigma, mu = 0.4, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-2, +2],
            u_range=[-2, +2]
        )

        gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        gauss_plane.set_fill_by_checkerboard( BLUE, opacity=0.3)


        gauss_plane_2 = Surface(
                    param_gauss,
                    resolution=(resolution_fa, resolution_fa),

                    v_range=[-np.sin(1), np.sin(1)],
                    u_range=[-np.cos(1), np.cos(1)]
                )

        gauss_plane_2.set_style(fill_opacity=1,stroke_color=RED)

        axes = ThreeDAxes()







class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 42
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u 
            y = v
            sigma, mu = 0.2, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4]
        )

        #gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        #gauss_plane.set_fill_by_checkerboard( BLUE, opacity=0.3)

        def param_gauss_2(u, v):
            x = u 
            y = v 
            sigma, mu = 0.2, [1, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane_2 = Surface(
            param_gauss_2,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4]
        )


        axes = ThreeDAxes()
        self.add(axes, gauss_plane)

        self.play(Transform(gauss_plane, gauss_plane_2))



class ThreeDSurfacePlot(ThreeDScene):
    def construct(self):
        resolution_fa = 42
        self.set_camera_orientation(phi=75 * DEGREES, theta=-30 * DEGREES)

        def param_gauss(u, v):
            x = u 
            y = v
            sigma, mu = 0.5, [0.0, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])

        gauss_plane = Surface(
            param_gauss,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4]
        )

        #gauss_plane.set_style(fill_opacity=1,stroke_color=GREEN)
        #gauss_plane.set_fill_by_checkerboard(BLUE, opacity=0.3)

        def param_gauss_2(u, v):
            x = u 
            y = v 
            sigma, mu = 0.5, [1, 0.0]
            d = np.linalg.norm(np.array([x - mu[0], y - mu[1]]))
            z = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
            return np.array([x, y, z])


        gauss_plane_2 = Surface(
            param_gauss_2,
            resolution=(resolution_fa, resolution_fa),
            v_range=[-4, +4],
            u_range=[-4, +4]
        )


        axes = ThreeDAxes()
        self.add(axes, gauss_plane)

        group = []

        for i in range(0, 10):
                    x, y = np.random.uniform(-2, 2, 2)
                    x, y, z = param_gauss(x, y)
                    dot = Dot3D(color=PURE_RED, radius=0.075).move_to(
                        np.array([x, y, z])
                    )
                    group.append(dot)

        anim_group = VGroup(*group)
        self.add(anim_group)

        self.wait(1)

        self.play(Transform(gauss_plane, gauss_plane_2))
    
        self.remove(anim_group)

        group_1 = []

        for i in range(0, 10):
                    x, y = np.random.uniform(-2, 2, 2)
                    x, y, z = param_gauss_2(x, y)
                    dot = Dot3D(color=PURE_RED, radius=0.075).move_to(
                        np.array([x, y, z])
                    )
                    group_1.append(dot)

        anim_group_1 = VGroup(*group_1)
        self.add(anim_group_1)
        self.wait(2)



    