from os import write
from manim import *
import numpy as np
from numpy import random
from numpy.linalg.linalg import norm
import common as cm
import copy
import math
import perlin_noise as pns


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


class FitDynamicLand(ThreeDScene):
    def construct(self):

        resolution_fa = 24
        res = 4

        def param_surf(u, v):
            x = u
            y = v
            z = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            return np.array([x, y, z * 0.2])

        my_plane = ParametricSurface(
            param_surf,
            resolution=(resolution_fa, resolution_fa),
            v_min=-res,
            v_max=+res,
            u_min=-res,
            u_max=+res,
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

        my_plane.set_fill_by_value(axes=axes, colors=list)

        self.add(axes)
        self.set_camera_orientation(phi=40 * DEGREES, theta=22 * DEGREES, distance=12)

        # Generate frame details
        main_title = Text("Generation n = ", size=0.75)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 3 + UP * 3.5)

        gen_num = Text("000", size=0.75)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = Text("Fitness", slant=ITALIC).scale(0.65).set_shade_in_3d(True)
        fit_text.move_to(LEFT * 5.3 + UP * 3)
        self.add_fixed_in_frame_mobjects(fit_text)

        x3d = Text("Trait X").scale(1)
        x3d.move_to(DOWN * 5)
        self.add(x3d)

        y3d = Text("Trait Y").scale(1)
        y3d.rotate(PI / 2)
        y3d.move_to(LEFT * 5)
        self.add(y3d)

        current = my_plane

        # MAIN LOOP
        max_gens = 4
        n_gen = 1
        pop_size = 20
        initialize = 0

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(-4, -3)
                    ran_y = np.random.uniform(-4, -3)
                    x, y, z = param_surf(ran_x, ran_y)
                    coord = x, y, z
                    features = coord, ind_num, z
                    my_individual = cm.make_ind(features)
                    cm.add_to_archive(my_individual, archive)

                # Stop inizialization
                initialize = 1

                self.add(current)

            if n_gen < max_gens + 1:

                # ANIMATION GROUP
                group = []

                for ind_in_archive in range(0, len(archive)):
                    x, y, z = archive[ind_in_archive].coord
                    dot = Dot3D(radius=0.075, color=[RED_D]).move_to(
                        np.array([x, y, z])
                    )
                    group.append(dot)

                # ANIMATION

                anim_group = VGroup(*group)
                self.add(anim_group)
                gen_succ = Text(str(n_gen), size=0.75)
                gen_num.become(gen_succ)
                gen_num.next_to(main_title, RIGHT)

                self.wait(1)
                self.remove(anim_group)

                # SELECTION

                # Extract fit (normalize) and id from Class Ind

                ind_id = np.fromiter(archive.keys(), dtype=int)
                ind_fit = np.zeros(shape=(len(archive), 1))

                # Update fitness from coordinates
                for i in range(0, len(archive)):
                    x, y, archive[i].fitness = archive[i].coord

                # Fitness to array
                for i in range(0, len(archive)):
                    # Add +1 to reduce differences in fitness after normalization
                    ind_fit[i] = archive[i].fitness

                # Before normalize check if are all the same (Function does not return NaN)
                if np.max(ind_fit) != np.min(ind_fit):
                    ind_fit_array = (ind_fit - ind_fit.min()) / (
                        ind_fit.max() - ind_fit.min()
                    )
                    norm_fit = ind_fit_array / ind_fit_array.sum()
                else:
                    # Same probability for each
                    norm_fit = np.full((len(ind_fit)), 1 / (len(ind_fit)))

                new_archive = copy.deepcopy(archive)
                new_pop_id = np.random.choice(
                    ind_id, len(norm_fit), p=norm_fit.flatten()
                )

                print(new_pop_id)
                print(len(archive))

                for j in range(0, len(new_archive)):
                    new_archive[j] = copy.deepcopy(archive[new_pop_id[j]])

                archive = copy.deepcopy(new_archive)

                cm.mutate_cauchy(archive, n_gen, 0.15)

                n_gen += 1

            else:
                return 1


class ThreeDTrial(ThreeDScene):
    def construct(self):
        resolution_fa = 45
        self.set_camera_orientation(
            phi=65 * DEGREES, theta=-105 * DEGREES, distance=155
        )

        res = 4

        def param_gauss_mod(u, v):
            x = u
            y = v
            z = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            return np.array([x + 0.25, y + 0.25, z * 0.28])

        g_plane = ParametricSurface(
            param_gauss_mod,
            **{"fill_color": BLUE_D, "shade_in_3d": True, "should_make_jagged": True},
            resolution=(resolution_fa, resolution_fa),
            v_min=-res,
            v_max=+res,
            u_min=-res,
            u_max=+res
            # checkerboard_colors= [BLUE_A, BLUE_E]
        )

        axis = ThreeDAxes(
            x_length=res * 2,
            y_length=res * 2,
            z_length=res,
            y_range=(0, 5, 1),
            x_range=(1, 5, 1),
            z_range=(0, 5, 1),
        )

        g_plane.set_style(fill_opacity=1, stroke_color=BLACK)
        g_plane.set_fill_by_checkerboard(BLUE_D, opacity=0.9)
        g_plane.set_shade_in_3d(True)

        axis.coords_to_point(0, 0, 0)

        self.add(axis)
        self.add(g_plane)
