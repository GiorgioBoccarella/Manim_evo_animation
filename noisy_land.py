from manim import *
import numpy as np
from numpy.linalg.linalg import norm
import common as cm
import copy
import perlin_noise as pns
from scipy.stats import kde
from perlin_noise import *

res = 3


def mutate_norm(archive_dict, prob):

    for ind_in_archive in range(0, len(archive_dict)):
        n, p = 1, prob  # number of trials, probability of each trial
        s = int(np.random.binomial(n, p, 1))
        if s > 0:
            x, y, z = archive_dict[ind_in_archive].coord
            mu, sigma = 0, 0.15  # mean and standard deviation
            x_ran = float(np.random.normal(mu, sigma, 1))
            y_ran = float(np.random.normal(mu, sigma, 1))
            x_m = x + x_ran
            y_m = y + y_ran
            # Check they do not move outside the u,v
            if -res < x_m < res and -res < y_m < res:
                archive_dict[ind_in_archive].coord = x_m, y_m, z
    return 1


p2 = PerlinNoiseFactory(2, tile=(1000, 0))

p3 = PerlinNoiseFactory(2, tile=(50, 50))

p4 = PerlinNoiseFactory(2, tile=(4, 80), unbias=True)

p5 = PerlinNoiseFactory(2, tile=(3, 3))


class SimPlot(ThreeDScene):
    def construct(self):

        # Initialize surface and labels

        surface = ParametricSurface(
            lambda u, v: np.array([u, v, 0]),
            resolution=60,
            u_min=-3,
            u_max=3,
            v_min=-3,
            v_max=3,
        )
        # self.move_camera(0.7*np.pi/2, 0.4 * np.pi)
        self.move_camera(phi=55 * DEGREES, theta=-65 * DEGREES)
        self.add(surface)

        axes = ThreeDAxes()
        self.add(axes)

        main_title = Text("Generation n = ", size=0.75)
        self.add_fixed_in_frame_mobjects(main_title)
        main_title.move_to(RIGHT * 3 + UP * 3.5)

        gen_num = Text("000", size=0.75)
        self.add_fixed_in_frame_mobjects(gen_num)
        gen_num.next_to(main_title, RIGHT)

        # Axes labels
        fit_text = Text("Dynamic", slant=ITALIC).scale(0.65).set_shade_in_3d(True)
        fit_text.move_to(LEFT * 5.3 + UP * 3)
        self.add_fixed_in_frame_mobjects(fit_text)

        x3d = Text("Trait X").scale(1)
        x3d.move_to(DOWN * 5)
        self.add(x3d)

        y3d = Text("Trait y").scale(1)
        y3d.rotate(PI / 2)
        y3d.move_to(LEFT * 5)
        self.add(y3d)

        # MAIN LOOP
        max_gens = 55
        n_gen = 1
        pop_size = 50
        initialize = 0

        while n_gen < max_gens + 1:

            z_rans = np.array([0.45, 0.45, 0.35, 0.35])

            # Generate individuals
            if initialize == 0:
                archive = {}

                for ind_num in range(0, pop_size):
                    ran_x = np.random.uniform(-2.7, -1.7)
                    ran_y = np.random.uniform(-2.7, -1.7)
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

                np.random.seed(n_gen)
                z_rans += np.random.normal(0, 0.08, 4)
                z_rans[z_rans < 0] = 0

                z_rans = np.round(z_rans / np.linalg.norm(z_rans, 1.0), 3)

                def perlin_surface(u, v, z_rans=z_rans, my_seed=n_gen):
                    x = u
                    y = v
                    np.random.seed(my_seed)

                    z_3 = p3.get_plain_noise(u + 51, v) * z_rans[0]
                    z_2 = p2.get_plain_noise(u, v) * z_rans[1]
                    z_4 = p4.get_plain_noise(u + 2, v + 3) * z_rans[2]
                    z_5 = p5.get_plain_noise(u + 10, v - 7) * z_rans[3]

                    d = np.sqrt(x * x + y * y)
                    sigma, mu = 0.4, 0.0
                    z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))

                    return [u, v, (z_2 + z_3 + z_4 + z_5) * 0.65 + z * 1.2]

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
                    dot = Dot3D(radius=0.055, color=[RED_D]).move_to(
                        np.array([archive[ind_in_archive].coord])
                    )
                    group.append(dot)

                # Move id and update surface
                surface.become(new_surface)
                anim_group = VGroup(*group)
                self.add(anim_group)

                # Text update
                gen_succ = Text(str(n_gen), size=0.75)
                gen_num.become(gen_succ)
                gen_num.next_to(main_title, RIGHT)

                self.wait(0.8)
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

                print(new_pop_id)
                print(len(archive))

                for j in range(0, len(new_archive)):
                    new_archive[j] = copy.deepcopy(archive[new_pop_id[j]])

                archive = copy.deepcopy(new_archive)

                mutate_norm(archive, 1)

                n_gen += 1

            else:
                return 1
