from os import write
from manim import *
import numpy as np
from numpy import random
from numpy.linalg.linalg import norm
import common as cm
import copy
import math
import perlin_noise as pns


class FitDynamicLand(ThreeDScene):
    def construct(self):

        resolution_fa = 24
        res = 4

        p2 = pns.PerlinNoiseFactory(2)

        def param_gauss_mod_find(u, v):
            x = u
            y = v
            z = p2.get_plain_noise(u, v)
            z_1 = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x **
                                                                    3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2)
            return (x, y, z)

        my_plane = ParametricSurface(
            param_gauss_mod_find,
            resolution=(resolution_fa, resolution_fa),
            v_min=-res,
            v_max=+res,
            u_min=-res,
            u_max=+res,
        )

        axes = ThreeDAxes()
        self.add(axes)

        current = my_plane

        self.move_camera(phi=55 * DEGREES, theta=-65*DEGREES)

        main_title = Text('Generation n = ', size=0.75)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 3 + UP * 3.5)

        gen_num = Text('000', size=0.75)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = Text("Fitness", slant=ITALIC).scale(
            0.65).set_shade_in_3d(True)
        fit_text.move_to(LEFT * 5.3 + UP * 3)
        self.add_fixed_in_frame_mobjects(fit_text)

        x3d = Text("Trait X").scale(1)
        x3d.move_to(DOWN*5)
        self.add(x3d)

        y3d = Text("Trait y").scale(1)
        y3d.rotate(PI/2)
        y3d.move_to(LEFT*5)
        self.add(y3d)

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
                    x, y, z = param_gauss_mod_find(ran_x, ran_y, n_gen)
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
                        np.array([x, y, z]))
                    group.append(dot)

                # ANIMATION

                def param_gauss_mod_find(u, v, dt=n_gen):
                    x = u
                    y = v
                    z = p2.get_plain_noise(u,v)
                    z_1 = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x **
                                                                         3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2)
                    return (x, y, z)

                new = ParametricSurface(param_gauss_mod_find, resolution=(resolution_fa, resolution_fa),
                                        v_min=-res,
                                        v_max=+res,
                                        u_min=-res,
                                        u_max=+res)

                current.become(new)
                #self.add(current)

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
                    ind_fit_array = (ind_fit - ind_fit.min()) / \
                        (ind_fit.max() - ind_fit.min())
                    norm_fit = (ind_fit_array / ind_fit_array.sum())
                else:
                    # Same probability for each
                    norm_fit = np.full((len(ind_fit)), 1/(len(ind_fit)))

                new_archive = copy.deepcopy(archive)
                new_pop_id = np.random.choice(
                    ind_id, len(norm_fit), p=norm_fit.flatten())

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
            phi=65 * DEGREES, theta=-105*DEGREES, distance=155)

        res = 4

        def param_gauss_mod(u, v):
            x = u
            y = v
            z = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x **
                                                               3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2)
            return np.array([x+0.25, y + 0.25, z*0.28])

        g_plane = ParametricSurface(
            param_gauss_mod,
            **{
                'fill_color': BLUE_D,
                'shade_in_3d': True,
                'should_make_jagged': True
            },
            resolution=(resolution_fa, resolution_fa),
            v_min=-res,
            v_max=+res,
            u_min=-res,
            u_max=+res
            # checkerboard_colors= [BLUE_A, BLUE_E]
        )

        axis = ThreeDAxes(x_length=res*2, y_length=res*2, z_length=res,
                          y_range=(0, 5, 1), x_range=(1, 5, 1), z_range=(0, 5, 1))

        g_plane.set_style(fill_opacity=1, stroke_color=BLACK)
        g_plane.set_fill_by_checkerboard(BLUE_D, opacity=0.9)
        g_plane.set_shade_in_3d(True)

        axis.coords_to_point(0, 0, 0)

        self.add(axis)
        self.add(g_plane)
