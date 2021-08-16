import math
import numpy as np
import perlin_noise as pns

p2 = pns.PerlinNoiseFactory(2)

res = 4

np.random.rand(4)

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
        
def param_gauss(u, v):
            x = u
            y = v
            d = np.sqrt(x * x + y * y)
            sigma, mu = 0.4, 0.0
            z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            return np.array([x , y, z*0.6])
        
        
def param_gauss_mod(u, v):
    x = u
    y = v
    z = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2) 
    return np.array([x+0.25 , y+ 0.25, z*0.28])



def param_gauss_mod_find(u, v, dt=1):
    np.random.seed(dt)
    x = u
    y = v
    z = p2.get_plain_noise(u, v) + np.random.uniform(0, 2)
    z_1 = 3*(1-x)**2.*math.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x **
                                                            3 - y**5)*math.exp(-x**2-y**2) - 1/3*math.exp(-(x+1)**2 - y**2)
    return (x, y, z*0.4 + z_1*0.3)
        
def my_func_comp(u, v):
            w = np.sin(.2 * np.cos(u ** 2) + np.sin(u) + np.cos(v))
            return np.array([u, v, (w)*1.2])
        
        
def my_func_comp_find(u, v):
            w = .1 * u ** 2 + np.sin(u) + np.cos(v) + 2
            return (u, v, (w -2.5)*0.45)
        
        
def find_z_from_coord(u, v):
        x = u
        y = v
        d = np.sqrt(x * x + y * y)
        sigma, mu = 0.4, 0.0
        z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        return (x, y, z*0.6)
    
def add_to_archive(individual, archive):
            archive[individual.id] = individual
            return 1
        
def mutate_uni(archive_dict, mutation_step):
                    
    for ind_in_archive in range(0, len(archive_dict)):
        x, y, z = archive_dict[ind_in_archive].coord
        x_ran = np.random.uniform(-mutation_step , mutation_step)
        y_ran = np.random.uniform(-mutation_step , mutation_step)
        x_m = x + x_ran
        y_m = y + y_ran
        # Check they do not move outside the u,v
        if -res < x_m < res and -res < y_m < res:
            new_x, new_y, new_z = my_func_comp_find(x_m, y_m)
            archive_dict[ind_in_archive].coord = new_x, new_y, new_z
    return 1

def mutate_cauchy(archive_dict, dt, prob):
                    
    for ind_in_archive in range(0, len(archive_dict)):
        n, p = 1, prob  # number of trials, probability of each trial
        s = int(np.random.binomial(n, p, 1))
        if s > 0:
            x, y, z = archive_dict[ind_in_archive].coord
            mu, sigma = 0, 0.47 # mean and standard deviation
            x_ran = float(np.random.normal(mu, sigma, 1))
            y_ran = float(np.random.normal(mu, sigma, 1))
            x_m = x + x_ran
            y_m = y + y_ran
            # Check they do not move outside the u,v
            if -res < x_m < res and -res < y_m < res:
                new_x, new_y, new_z = param_gauss_mod_find(x_m, y_m, dt)
                archive_dict[ind_in_archive].coord = new_x, new_y, new_z
    return 1

def mutate_2(archive_dict, prob):
                    
    for ind_in_archive in range(0, len(archive_dict)):
        n, p = 1, prob  # number of trials, probability of each trial
        s = int(np.random.binomial(n, p, 1))
        if s > 0:
            x, y, z = archive_dict[ind_in_archive].coord
            mu, sigma = 0, 0.95 # mean and standard deviation
            x_ran = float(np.random.normal(mu, sigma, 1))
            y_ran = float(np.random.normal(mu, sigma, 1))
            x_m = x + x_ran
            y_m = y + y_ran
            # Check they do not move outside the u,v
            if -res < x_m < res and -res < y_m < res:
                new_x, new_y, new_z = param_gauss_mod_find(x_m, y_m)
                archive_dict[ind_in_archive].coord = new_x, new_y, new_z
    return 1






