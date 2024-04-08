import torch

from vehicle_Isaacgym.vehicle.Terrains.Terrain import *

class Perlin(Terrain):
    def __init__(self, cfg, num_robots):
        super(Perlin, self).__init__(cfg=cfg, num_robots=num_robots)
        num_robots_per_map = int(num_robots / self.env_cols)
        left_over = num_robots % self.env_cols
        terrain = SubTerrain("terrain", width=self.width_per_env_pixels * self.env_rows,
                             length=self.width_per_env_pixels*self.env_cols,
                             vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
        seed = 20
        self.generate_perlin_noise_terrain(terrain, int(seed))
        for j in range(self.env_cols):
            for i in range(self.env_rows):

                difficulty = i / self.env_rows
                choice = j / self.env_cols
                # seed = difficulty * 20
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.length_per_env_pixels
                end_y = self.border + (j + 1) * self.length_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw[start_x-200:end_x-200, start_y-200:end_y-200]

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length  # - 为变化出生点x
                env_origin_y = (j + 0.5) * self.env_width

                x1 = int((self.env_length / 2 - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2 + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2 - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2 + 1) / self.horizontal_scale)
                env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
                # env_origin_z=0.
                self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        self.addheightfield()


    def generate_perlin_noise_terrain(self, terrain, seed=0):
        x = np.linspace(0, terrain.width*terrain.horizontal_scale, terrain.width)/10
        y = np.linspace(0, terrain.length*terrain.horizontal_scale, terrain.length)/10   #改变地形粗糙度
        X, Y = np.meshgrid(y, x)
        # Z=perlin(x,y,seed)
        Z=perlin(X,Y,seed)
        Z = Z.reshape(X.shape)/terrain.vertical_scale     #放大十倍
        terrain.height_field_raw+=Z.astype(terrain.height_field_raw.dtype)
        return Z

def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(2560, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
    return lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y
