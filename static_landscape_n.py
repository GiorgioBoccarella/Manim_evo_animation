from manim import *
import common as cm
import numpy as np
from numpy.linalg.linalg import norm
import copy

# import perlin_noise as pns
# from scipy.stats import kde
import math


# Seed of the simulation
np.random.seed(cm.params_sim["seed"])

# Generate color gradient for surface
my_list = cm.create_color_list()


class Sim(ThreeDScene):
    def construct(self):

        # Initialize surface and labels
        surf_res = cm.params_sim["res"]

        def param_surf(u, v):
            x = u
            y = v
            z = (
                3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
            )
            d = np.sqrt((u + 0.4) * (u + 0.4) + (v + 0.4) * (v + 0.4))
            sigma, mu = 0.7, 0.1
            z_1 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

            return np.array([x, y, z * 0.14 + z_1 * 0.8])

        surface = ParametricSurface(
            param_surf,
            resolution=65,
            u_min=-surf_res,
            u_max=surf_res,
            v_min=-surf_res,
            v_max=surf_res,
        )

        axis = ThreeDAxes(
            x_length=(6),
            y_length=(6),
            z_length=(1.3),
            x_range=(0, 6, 1),
            y_range=(0, 6, 1),
            z_range=(0, 1, 1),
        )

        # self.move_camera(0.7*np.pi/2, 0.4 * np.pi)

        # Camera and Labels
        self.set_camera_orientation(phi=55 * DEGREES, theta=42 * DEGREES, distance=90)
        self.play(Create(axis))
        surface.set_fill_by_value(axes=axis, colors=my_list)
        self.play(Create(surface))
        self.play(Create(axis))

        main_title = Text("Generation n = ", size=0.50)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 4.2 + UP * 3.5)

        gen_num = Text("000", size=0.6)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = (
            Text("Static landscape", slant=ITALIC).scale(0.4).set_shade_in_3d(True)
        )
        fit_text.move_to(LEFT * 5.3 + UP * 3)
        self.add_fixed_in_frame_mobjects(fit_text)

        sel_text = Text("selection", slant=ITALIC).scale(0.4).set_shade_in_3d(True)
        sel_text.move_to(LEFT * 5.2 + UP * 2.7)
        self.add_fixed_in_frame_mobjects(sel_text)

        x3d = Text("Trait X").scale(0.65)
        x3d.move_to(DOWN * 3.5)
        self.add(x3d)

        y3d = Text("Trait Y").scale(0.65)
        y3d.rotate(PI / 2)
        y3d.move_to(LEFT * 3.5)
        self.add(y3d)

        # MAIN LOOP
        max_gens = cm.params_sim["max_gen"]
        pop_size = cm.params_sim["pop_size"]
        n_gen = 1
        initialize = 0
        coord_array = np.zeros([2, pop_size])

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(1.8, 2.4)
                    ran_y = np.random.uniform(-0.4, 0.4)
                    x, y, z = ran_x, ran_y, 0
                    coord = x, y, z
                    features = coord, ind_num, z
                    my_individual = cm.make_ind(features)
                    cm.add_to_archive(my_individual, archive)

                initialize = 1

            if n_gen < max_gens + 1:

                # Generate landscape (FD) based of id coord
                # Environments changes based on new coordinates

                group = []

                for ind_in_archive in range(0, len(archive)):
                    x_1, y_1, z_1 = archive[ind_in_archive].coord
                    # This calculates new fitness necesary for position

                    x, y, z = param_surf(x_1, y_1)

                    archive[ind_in_archive].coord = (
                        x,
                        y,
                        # This is the fitness
                        z,
                    )
                    # Add to group
                    dot = cm.MyInd(color=PURE_RED, radius=0.075).move_to(
                        np.array([archive[ind_in_archive].coord])
                    )
                    group.append(dot)

                # Move id and update surface

                anim_group = VGroup(*group)
                self.add(anim_group)

                # Text update
                gen_succ = Text(str(n_gen), size=0.55)
                gen_num.become(gen_succ)
                gen_num.next_to(main_title, RIGHT)

                self.begin_ambient_camera_rotation(rate=0.06, about="theta")
                self.wait(0.25)
                self.stop_ambient_camera_rotation()
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
                # This is actually doing the selection by weighted lottery
                new_pop_id = np.random.choice(
                    ind_id, len(norm_fit), p=norm_fit.flatten()
                )

                print(new_pop_id)
                print(len(archive))

                for j in range(0, len(new_archive)):
                    new_archive[j] = copy.deepcopy(archive[new_pop_id[j]])

                archive = copy.deepcopy(new_archive)

                cm.mutate_norm(archive, cm.params_sim["mut_rate"])

                n_gen += 1

            else:
                return 1
