"""NOTE: The Perlin noise implementation is from
   https://gist.github.com/eevee/26f547457522755cb1fb8739d0ea89a1"""

"""Perlin noise implementation."""
# Licensed under ISC



from manim import *
from itertools import product
import math
import random

def smoothstep(t):
    """Smooth curve with a zero derivative at 0 and 1, making it useful for
    interpolating.
    """
    return t * t * (3. - 2. * t)


def lerp(t, a, b):
    """Linear interpolation between a and b, given a fraction t."""
    return a + t * (b - a) 


class PerlinNoiseFactory(object):
    """Callable that produces Perlin noise for an arbitrary point in an
    arbitrary number of dimensions.  The underlying grid is aligned with the
    integers.
    There is no limit to the coordinates used; new gradients are generated on
    the fly as necessary.
    """

    def __init__(self, dimension, octaves=1, tile=(), unbias=False):
        """Create a new Perlin noise factory in the given number of dimensions,
        which should be an integer and at least 1.
        More octaves create a foggier and more-detailed noise pattern.  More
        than 4 octaves is rather excessive.
        ``tile`` can be used to make a seamlessly tiling pattern.  For example:
            pnf = PerlinNoiseFactory(2, tile=(0, 3))
        This will produce noise that tiles every 3 units vertically, but never
        tiles horizontally.
        If ``unbias`` is true, the smoothstep function will be applied to the
        output before returning it, to counteract some of Perlin noise's
        significant bias towards the center of its output range.
        """
        self.dimension = dimension
        self.octaves = octaves
        self.tile = tile + (0,) * dimension
        self.unbias = unbias

        # For n dimensions, the range of Perlin noise is ±sqrt(n)/2; multiply
        # by this to scale to ±1
        self.scale_factor = 2 * dimension ** -0.5

        self.gradient = {}

    def _generate_gradient(self):
        # Generate a random unit vector at each grid point -- this is the
        # "gradient" vector, in that the grid tile slopes towards it

        # 1 dimension is special, since the only unit vector is trivial;
        # instead, use a slope between -1 and 1
        if self.dimension == 1:
            return (random.uniform(-1, 1),)

        # Generate a random point on the surface of the unit n-hypersphere;
        # this is the same as a random unit vector in n dimensions.  Thanks
        # to: http://mathworld.wolfram.com/SpherePointPicking.html
        # Pick n normal random variables with stddev 1
        random_point = [random.gauss(0, 1) for _ in range(self.dimension)] 
        # Then scale the result to a unit vector
        scale = sum(n * n for n in random_point) ** -0.5 
        return tuple(coord * scale for coord in random_point)

    def get_plain_noise(self, *point):
        """Get plain noise for a single point, without taking into account
        either octaves or tiling.
        """
        if len(point) != self.dimension:
            raise ValueError("Expected {} values, got {}".format(
                self.dimension, len(point)))

        # Build a list of the (min, max) bounds in each dimension
        grid_coords = []
        for coord in point:
            min_coord = math.floor(coord)
            max_coord = min_coord + 1
            grid_coords.append((min_coord, max_coord))

        # Compute the dot product of each gradient vector and the point's
        # distance from the corresponding grid point.  This gives you each
        # gradient's "influence" on the chosen point.
        dots = []
        for grid_point in product(*grid_coords):
            if grid_point not in self.gradient:
                self.gradient[grid_point] = self._generate_gradient()
            gradient = self.gradient[grid_point]

            dot = 0
            for i in range(self.dimension):
                dot += gradient[i] * (point[i] - grid_point[i])
            dots.append(dot)

        # Interpolate all those dot products together.  The interpolation is
        # done with smoothstep to smooth out the slope as you pass from one
        # grid cell into the next.
        # Due to the way product() works, dot products are ordered such that
        # the last dimension alternates: (..., min), (..., max), etc.  So we
        # can interpolate adjacent pairs to "collapse" that last dimension.  Then
        # the results will alternate in their second-to-last dimension, and so
        # forth, until we only have a single value left.
        dim = self.dimension
        while len(dots) > 1:
            dim -= 1
            s = smoothstep(point[dim] - grid_coords[dim][0])

            next_dots = []
            while dots:
                next_dots.append(lerp(s, dots.pop(0), dots.pop(0)))

            dots = next_dots

        return dots[0] * self.scale_factor

    def __call__(self, *point):
        """Get the value of this Perlin noise function at the given point.  The
        number of values given should match the number of dimensions.
        """
        ret = 0
        for o in range(self.octaves):
            o2 = 1 << o
            new_point = []
            for i, coord in enumerate(point):
                coord *= o2
                if self.tile[i]:
                    coord %= self.tile[i] * o2
                new_point.append(coord)
            ret += self.get_plain_noise(*new_point) / o2

        # Need to scale n back down since adding all those extra octaves has
        # probably expanded it beyond ±1
        # 1 octave: ±1
        # 2 octaves: ±1½
        # 3 octaves: ±1¾
        ret /= 2 - 2 ** (1 - self.octaves )

        if self.unbias:
            # The output of the plain Perlin noise algorithm has a fairly
            # strong bias towards the center due to the central limit theorem
            # -- in fact the top and bottom 1/8 virtually never happen.  That's
            # a quarter of our entire output range!  If only we had a function
            # in [0..1] that could introduce a bias towards the endpoints...
            r = (ret + 1) / 2
            # Doing it this many times is a completely made-up heuristic.
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smoothstep(r) 
            ret = r * 2 - 1

        return ret


"""My own code for the terrain."""

class PerlinNoiseFactory_1(object):
    def __init__(self, dimension, octaves=1, tile=(), unbias=False, seed=1):
        self.dimension = dimension
        self.octaves = octaves
        self.tile = tile + (0,) * dimension
        self.unbias = unbias
        self.scale_factor = 2 * dimension ** -0.5
        self.random = random
        self.random.seed(seed)
        self.seed = seed

        self.gradient = {}

    def _generate_gradient(self):
        if self.dimension == 1:
            return (self.random.uniform(-1, 1),)
        random_point = [self.random.gauss(0, 1) for _ in range(self.dimension)]
        scale = sum(n * n for n in random_point) ** -0.5
        return tuple(coord * scale for coord in random_point)

    def get_plain_noise(self, *point):
        if len(point) != self.dimension:
            raise ValueError("Expected {} values, got {}".format(
                self.dimension, len(point)))
        grid_coords = []
        for coord in point:
            min_coord = math.floor(coord)
            max_coord = min_coord + 1
            grid_coords.append((min_coord, max_coord))
        dots = []
        for grid_point in product(*grid_coords):
            if grid_point not in self.gradient:
                self.gradient[grid_point] = self._generate_gradient()
            gradient = self.gradient[grid_point]

            dot = 0
            for i in range(self.dimension):
                dot += gradient[i] * (point[i] - grid_point[i])
            dots.append(dot)
        dim = self.dimension
        while len(dots) > 1:
            dim -= 1
            s = smoothstep(point[dim] - grid_coords[dim][0])
            next_dots = []
            while dots:
                next_dots.append(lerp(s, dots.pop(0), dots.pop(0)))
            dots = next_dots

        return dots[0] * self.scale_factor

    def __call__(self, *point):
        ret = 0
        for o in range(self.octaves):
            o2 = 1 << o
            new_point = []
            for i, coord in enumerate(point):
                coord *= o2
                if self.tile[i]:
                    coord %= self.tile[i] * o2
                new_point.append(coord)
            ret += self.get_plain_noise(*new_point) / o2
        ret /= 2 - 2 ** (1 - self.octaves)

        if self.unbias:
            r = (ret + 1) / 2
            for _ in range(int(self.octaves / 2 + 0.5)):
                r = smoothstep(r)
            ret = r * 2 - 1

        return ret






p2 = PerlinNoiseFactory(2, tile=(1000,0))

p3 = PerlinNoiseFactory(2, tile=(50,50))

p4 = PerlinNoiseFactory(2, tile=(4,80), unbias=True)

p5 = PerlinNoiseFactory(2, tile=(3,3))



class PerlinTerrain(ThreeDScene):
    def construct(self):
        surface = ParametricSurface(lambda u, v: np.array([u, v, 0]), 
                                    resolution= 80,
                                    u_min=-4,
                                    u_max=4,
                                    v_min=-4,
                                    v_max=4,
                                    )
        self.move_camera(0.7*np.pi/2, 0.4 * np.pi)
        self.add(surface)
        # self.begin_ambient_camera_rotation(rate=0.1)
        # Rotate for 5 seconds
        #self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait()
        z_rans = np.array([0.25, 0.25, 0.25, 0.35])
        for i in range(1,60):
            np.random.seed(i)
            z_rans += np.random.normal(0, 0.1, 4)
            z_rans[z_rans < 0] = 0
            
            z_rans = np.round(z_rans/np.linalg.norm(z_rans,1.0),3)
            
            print(z_rans)
            

            
            def perlin_surface(u, v, z_rans = z_rans,  my_seed=i):
                x = u
                y = v
                np.random.seed(my_seed)

                z_3 = p3.get_plain_noise(u + 51,v) * z_rans[0] 
                z_2 = p2.get_plain_noise(u,v) * z_rans[1] 
                z_4 = p4.get_plain_noise(u+2,v+3) * z_rans[2]
                z_5 = p5.get_plain_noise(u+10,v -7) * z_rans[3]
                
                d = np.sqrt(x * x + y * y)
                sigma, mu = 0.4, 0.0
                z = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            
                return [u, v, (z_2 + z_3 + z_4 + z_5) *0.4 + z] 
            
            surface_r = ParametricSurface(perlin_surface,
                                    resolution=80,
                                    u_min=-4,
                                    u_max=4,
                                    v_min=-4,
                                    v_max=4)
            surface.become(surface_r)
            self.wait()
    
    

