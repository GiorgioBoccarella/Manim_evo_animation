from manim import *
import numpy as np
from numpy.linalg.linalg import norm
import common as cm
import copy
import perlin_noise as pns
from scipy.stats import kde
import scipy.stats
from manim.mobject.three_dimensions import MyInd
import math

np.random.seed(123456)

my_list = cm.create_color_list()


class SimPlot3(ThreeDScene):
    def construct(self):

        # Initialize surface and labels

        surf_res = 3

        surface = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            resolution=65,
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
        )

        # self.move_camera(0.7*np.pi/2, 0.4 * np.pi)

        # Camera and Labels
        self.set_camera_orientation(phi=55 * DEGREES, theta=42 * DEGREES, distance=90)
        surface.set_fill_by_value(axes=axes, colors=my_list)
        self.play(Create(surface))
        self.play(Create(axes))

        main_title = Text("Generation n = ", size=0.50)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 5 + UP * 3.5)

        gen_num = Text("000", size=0.55)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = (
            Text("Frequency-dependent", slant=ITALIC).scale(0.4).set_shade_in_3d(True)
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
        max_gens = 130
        n_gen = 1
        pop_size = 140
        initialize = 0
        coord_array = np.zeros([2, pop_size])

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(1.9, 2.4)
                    ran_y = np.random.uniform(1.9, 2.4)
                    x, y, z = ran_x, ran_y, 0
                    coord = x, y, z
                    features = coord, ind_num, z
                    my_individual = cm.make_ind(features)
                    cm.add_to_archive(my_individual, archive)

                initialize = 1

            if n_gen < max_gens + 1:

                # Generate landscape (FD) based of id coord
                # Environments changes based on new coordinates
                for i in range(0, len(archive)):
                    x, y, z = archive[i].coord
                    coord_array[0][i] = x
                    coord_array[1][i] = y

                group = []

                k = kde.gaussian_kde(coord_array)

                def kde_calc(u, v):
                    # KDE minus the gaussian
                    x = (u, v)
                    z = k(x)
                    d = np.sqrt(u * u + v * v)
                    sigma, mu = 0.4, 0.0
                    z_1 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

                    x_mod = u - 0.9
                    y_mod = v + 0.9
                    d_1 = np.sqrt(x_mod * x_mod + y_mod * y_mod)
                    z_2 = np.exp(-((d_1 - mu) ** 2 / (2.0 * sigma ** 2)))

                    x_mod1 = u - 0.9
                    y_mod1 = v - 0.9
                    d_11 = np.sqrt(x_mod1 * x_mod1 + y_mod1 * y_mod1)
                    z_21 = np.exp(-((d_11 - mu) ** 2 / (2.0 * sigma ** 2)))

                    arr = np.array(
                        [
                            u,
                            v,
                            (-z * 0.28 + z_1 * 1.15 + z_2 * 1.15 + z_21 * 1.15),
                        ]
                    )

                    return arr

                for ind_in_archive in range(0, len(archive)):
                    x, y, z = archive[ind_in_archive].coord
                    coord = (x, y)
                    # This calculates new fitness necesary for position
                    z_mod = k(coord) * -0.28

                    d = np.sqrt(x * x + y * y)
                    sigma, mu = 0.4, 0.0
                    z_1 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

                    x_mod = x - 0.9
                    y_mod = y + 0.9
                    d_1 = np.sqrt(x_mod * x_mod + y_mod * y_mod)
                    z_2 = np.exp(-((d_1 - mu) ** 2 / (2.0 * sigma ** 2)))

                    x_mod1 = x - 0.9
                    y_mod1 = y - 0.9
                    d_11 = np.sqrt(x_mod1 * x_mod1 + y_mod1 * y_mod1)
                    z_21 = np.exp(-((d_11 - mu) ** 2 / (2.0 * sigma ** 2)))

                    archive[ind_in_archive].coord = (
                        x,
                        y,
                        # This is the fitness
                        z_mod[0] + z_1 * 1.15 + z_2 * 1.15 + z_21 * 1.15,
                    )
                    # Add to group
                    dot = MyInd(color=PURE_RED, radius=0.075).move_to(
                        np.array([archive[ind_in_archive].coord])
                    )
                    group.append(dot)

                coord_array = np.zeros([2, pop_size])

                for i in range(0, len(archive)):
                    x, y, z = archive[i].coord
                    coord_array[0][i] = x
                    coord_array[1][i] = y

                new_surface = ParametricSurface(
                    kde_calc,
                    resolution=65,
                    u_min=-3,
                    u_max=3,
                    v_min=-3,
                    v_max=3,
                    fill_opacity=0.85,
                )

                # Move id and update surface
                new_surface.set_fill_by_value(axes=axes, colors=my_list)
                # Only first animation is a transform
                if n_gen == 1:
                    self.play(Transform(surface, new_surface))
                surface.become(new_surface)
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

                cm.mutate_norm(archive, 0.6)

                n_gen += 1

            else:
                return 1


class SimPlot5(ThreeDScene):
    def construct(self):

        # Initialize surface and labels

        surf_res = 3

        surface = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            resolution=65,
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
        )

        # self.move_camera(0.7*np.pi/2, 0.4 * np.pi)

        # Camera and Labels
        self.set_camera_orientation(phi=55 * DEGREES, theta=42 * DEGREES, distance=90)
        surface.set_fill_by_value(axes=axes, colors=my_list)
        self.play(Create(surface))
        self.play(Create(axes))

        main_title = Text("Generation n = ", size=0.50)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 5 + UP * 3.5)

        gen_num = Text("000", size=0.55)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = (
            Text("Frequency-dependent", slant=ITALIC).scale(0.4).set_shade_in_3d(True)
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
        max_gens = 130
        n_gen = 1
        pop_size = 140
        initialize = 0
        coord_array = np.zeros([2, pop_size])

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(-1.7, -2.3)
                    ran_y = np.random.uniform(-1.7, -2.3)
                    x, y, z = ran_x, ran_y, 0
                    coord = x, y, z
                    features = coord, ind_num, z
                    my_individual = cm.make_ind(features)
                    cm.add_to_archive(my_individual, archive)

                initialize = 1

            if n_gen < max_gens + 1:

                # Generate landscape (FD) based of id coord
                # Environments changes based on new coordinates
                for i in range(0, len(archive)):
                    x, y, z = archive[i].coord
                    coord_array[0][i] = x
                    coord_array[1][i] = y

                group = []

                k = kde.gaussian_kde(coord_array)

                def kde_calc(u, v):
                    # KDE minus the gaussian
                    x = (u, v)
                    z = k(x)

                    x = u
                    y = v
                    z_1 = (
                        3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                        - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                        - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
                    )

                    arr = np.array(
                        [
                            u,
                            v,
                            (-z * 0.3 + z_1 * 0.2),
                        ]
                    )

                    return arr

                for ind_in_archive in range(0, len(archive)):
                    x, y, z = archive[ind_in_archive].coord
                    coord = (x, y)
                    # This calculates new fitness necesary for position
                    z_mod = k(coord) * -0.3

                    z_1 = (
                        3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
                        - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
                        - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
                    )

                    archive[ind_in_archive].coord = (
                        x,
                        y,
                        # This is the fitness
                        z_mod[0] + z_1 * 0.2,
                    )
                    # Add to group
                    dot = MyInd(color=PURE_RED, radius=0.075).move_to(
                        np.array([archive[ind_in_archive].coord])
                    )
                    group.append(dot)

                coord_array = np.zeros([2, pop_size])

                for i in range(0, len(archive)):
                    x, y, z = archive[i].coord
                    coord_array[0][i] = x
                    coord_array[1][i] = y

                new_surface = ParametricSurface(
                    kde_calc,
                    resolution=65,
                    u_min=-3,
                    u_max=3,
                    v_min=-3,
                    v_max=3,
                    fill_opacity=0.85,
                )

                # Move id and update surface
                new_surface.set_fill_by_value(axes=axes, colors=my_list)
                # Only first animation is a transform
                if n_gen == 1:
                    self.play(Transform(surface, new_surface))
                surface.become(new_surface)
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

                cm.mutate_norm(archive, 0.5)

                n_gen += 1

            else:
                return 1
