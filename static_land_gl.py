from os import write
from manim import *
import numpy as np
from numpy import random
from numpy.linalg.linalg import norm
import common as cm
import copy
import math
import perlin_noise as pns
from manim.opengl import *


class FitDynamicLand(Scene):
    def construct(self):

        resolution_fa = 24
        res = 4

        p2 = pns.PerlinNoiseFactory(2)

        def param_gauss_mod_find(u, v):
            x = u
            y = v
            z_noise = p2.get_plain_noise(u, v)
            z_f = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            return (x, y, z_f)

        my_plane = OpenGLSurface(
            param_gauss_mod_find,
            resolution=(resolution_fa, resolution_fa),
            v_min=-res,
            v_max=+res,
            u_min=-res,
            u_max=+res,
        )

        current = my_plane

        # MAIN LOOP
        max_gens = 4
        n_gen = 1
        pop_size = 20
        initialize = 0

        self.interactive_embed()

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(-4, -3)
                    ran_y = np.random.uniform(-4, -3)
                    x, y, z = param_gauss_mod_find(ran_x, ran_y)
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
                    dot = OpenGLSphere(radius=0.075, color=RED).move_to(
                        np.array([x, y, z])
                    )
                    group.append(dot)

                # ANIMATION

                def param_gauss_mod_find(u, v, dt=n_gen):
                    x = u
                    y = v
                    z = p2.get_plain_noise(u, v)
                    z_1 = (
                        3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                        - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                        - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
                    )
                    return (x, y, z_1)

                new = OpenGLSurface(
                    param_gauss_mod_find,
                    resolution=(resolution_fa, resolution_fa),
                    v_min=-res,
                    v_max=+res,
                    u_min=-res,
                    u_max=+res,
                    color=BLUE,
                )

                current.become(new)
                # self.add(current)

                anim_group = OpenGLGroup(*group)
                self.add(anim_group)

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
