from manim import *
import numpy as np
from numpy.linalg.linalg import norm
import copy
from scipy.stats import kde
from perlin_noise import *
import common as cm
from manim.mobject.three_dimensions import MyInd


# Seed of the simulation
np.random.seed(cm.params_sim["seed"])

# Generate color gradient for surface
my_list = cm.create_color_list()


# Generate 4 base noise structure for noisy landscape
p2 = PerlinNoiseFactory(2, tile=(10300, 0), unbias=True)

p3 = PerlinNoiseFactory(2, tile=(27, 50))

p4 = PerlinNoiseFactory(2, tile=(405, 80), unbias=True)

p5 = PerlinNoiseFactory(2, tile=(5, 3))


class SimPlot(ThreeDScene):
    def construct(self):

        # Initialize surface and labels
        surf_res = cm.params_sim["res"]

        surface = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            resolution=60,
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

        # Camera and Labels
        self.set_camera_orientation(phi=55 * DEGREES, theta=42 * DEGREES, distance=90)
        surface.set_fill_by_value(axes=axes, colors=my_list)
        self.play(Create(surface))
        self.play(Create(axes))

        main_title = Text("Generation n = ", size=0.50)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 4.2 + UP * 3.5)

        gen_num = Text("000", size=0.6)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = (
            Text("Dynamic landscape", slant=ITALIC).scale(0.4).set_shade_in_3d(True)
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

        z_rans = np.array([0.15, 0.25, 0.15, 0.25])

        while n_gen < max_gens + 1:

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(1.7, 2.3)
                    ran_y = np.random.uniform(1.7, 2.3)
                    x, y, z = ran_x, ran_y, 0
                    coord = x, y, z
                    features = coord, ind_num, z
                    my_individual = cm.make_ind(features)
                    cm.add_to_archive(my_individual, archive)

                # Stop inizialization
                initialize = 1

            if n_gen < max_gens + 1:

                # Generate landscape (FD) based of id coord
                # Environments changes based on new coordinates
                for i in range(0, len(archive)):
                    x, y, z = archive[i].coord
                    coord_array[0][i] = x
                    coord_array[1][i] = y

                group = []

                # Change env proportions

                s = int(np.random.binomial(1, 0.55, 1))
                if s > 0:
                    z_rans += np.random.normal(0, 0.14, 4)
                    z_rans[z_rans <= 0] = 0.001
                    z_rans = np.round(z_rans / z_rans.sum(), 4)

                def perlin_surface(u, v, z_rans=z_rans, my_seed=n_gen):
                    x = u
                    y = v
                    np.random.seed(my_seed)

                    z_3 = p3.get_plain_noise(u + 51, v) * z_rans[0]
                    z_2 = p2.get_plain_noise(u, v) * z_rans[1]
                    z_4 = p4.get_plain_noise(u + 246, v + 3) * z_rans[2]
                    z_5 = p5.get_plain_noise(u + 1500, v - 7) * z_rans[3]

                    d = np.sqrt(x * x + y * y)
                    sigma, mu = 0.45, 0.0
                    z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

                    return [u, v, (z_2 + z_3 + z_4 + z_5) * 0.47 + z * 1.3]

                new_surface = ParametricSurface(
                    perlin_surface,
                    resolution=60,
                    u_min=-3,
                    u_max=3,
                    v_min=-3,
                    v_max=3,
                )

                # Update id position to new coordinates and add to group
                group = []

                for ind_in_archive in range(0, len(archive)):
                    x, y, z = archive[ind_in_archive].coord
                    x, y, z = perlin_surface(x, y, z_rans=z_rans, my_seed=n_gen)
                    archive[ind_in_archive].coord = x, y, z
                    # Add to group
                    dot = MyInd(color=PURE_RED, radius=0.075).move_to(
                        np.array([archive[ind_in_archive].coord])
                    )
                    group.append(dot)

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
                new_pop_id = np.random.choice(
                    ind_id, len(norm_fit), p=norm_fit.flatten()
                )

                # print just to see how selection  is going
                print(new_pop_id)
                # print(len(archive))

                for j in range(0, len(new_archive)):
                    new_archive[j] = copy.deepcopy(archive[new_pop_id[j]])

                archive = copy.deepcopy(new_archive)

                cm.mutate_norm(archive, cm.params_sim["mut_rate"])

                n_gen += 1

            else:
                return 1
