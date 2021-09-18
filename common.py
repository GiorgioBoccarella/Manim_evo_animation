import math
import manim
import numpy as np


params_sim = {
    "res": 3,
    "seed": 123,
    "max_gen": 130,
    "pop_size": 100,
    "perlin_seed": 12345,
}

# Set seed for mutation function in common
np.random.rand(params_sim["seed"])


class Ind:
    def __init__(self, coord, id, fitness):
        self.coord = coord
        self.id = id
        self.fitness = fitness


def make_ind(features):
    coord, id, fit = features
    return Ind(coord, id, fit)


def add_to_archive(individual, archive):
    archive[individual.id] = individual
    return 1


def mutate_norm(archive_dict, prob):

    for ind_in_archive in range(0, len(archive_dict)):
        res = params_sim["res"]
        n, p = 1, prob  # number of trials, probability of each trial
        s = int(np.random.binomial(n, p, 1))
        if s > 0:
            x, y, z = archive_dict[ind_in_archive].coord
            mu, sigma = 0, 0.15  # mean and standard deviation
            x_ran = float(np.random.normal(mu, sigma, 1))
            y_ran = float(np.random.normal(mu, sigma, 1))
            # x_ran = float(scipy.stats.cauchy.rvs(loc=0, scale=0.025, size=1))
            # y_ran = float(scipy.stats.cauchy.rvs(loc=0, scale=0.025, size=1))
            x_m = x + x_ran
            y_m = y + y_ran
            # Check they do not move outside the u,v
            if -res < x_m < res and -res < y_m < res:
                archive_dict[ind_in_archive].coord = x_m, y_m, z
    return 1


def create_color_list():

    list = []

    my_res = 1000
    my_u_bound = 0.55
    my_l_bound = -0.15

    interval_1 = np.linspace(my_l_bound, -0.0000001, num=my_res)
    interval_2 = np.linspace(-0.0000001, my_u_bound, num=my_res)

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
                    manim.rgb_to_color(
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
                    manim.rgb_to_color(
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

    my_list = list

    return my_list


def param_gauss(u, v):
    x = u
    y = v
    d = np.sqrt(x * x + y * y)
    sigma, mu = 0.4, 0.0
    z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return np.array([x, y, z * 0.6])


def param_multi_mod(u, v):
    x = u
    y = v
    z = (
        3 * (1 - x) ** 2.0 * math.exp(-(x ** 2) - (y + 1) ** 2)
        - 10 * (x / 5 - x ** 3 - y ** 5) * math.exp(-(x ** 2) - y ** 2)
        - 1 / 3 * math.exp(-((x + 1) ** 2) - y ** 2)
    )
    return np.array([x + 0.25, y + 0.25, z * 0.28])


def my_func_comp(u, v):
    w = np.sin(0.2 * np.cos(u ** 2) + np.sin(u) + np.cos(v))
    return np.array([u, v, (w) * 1.2])


def my_func_comp_find(u, v):
    w = 0.1 * u ** 2 + np.sin(u) + np.cos(v) + 2
    return (u, v, (w - 2.5) * 0.45)
