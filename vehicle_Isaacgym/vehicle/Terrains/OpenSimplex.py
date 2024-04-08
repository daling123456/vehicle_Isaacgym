import torch
from opensimplex import OpenSimplex
from vehicle_Isaacgym.vehicle.Terrains.Terrain import *

class Open_Simplex(Terrain):
    def __init__(self, cfg, num_robots):
        super(Open_Simplex, self).__init__(cfg=cfg, num_robots=num_robots)
        num_robots_per_map = int(num_robots / self.env_cols)
        left_over = num_robots % self.env_cols
        terrain = SubTerrain("terrain", width=self.width_per_env_pixels * self.env_rows,
                             length=self.width_per_env_pixels*self.env_cols,
                             vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
        seed = 20
        self.generate_OpenSimplex_noise_terrain(terrain, int(seed))
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


    def generate_OpenSimplex_noise_terrain(self, terrain, seed=0):
        noise_gen= OpenSimplex(seed)
        x = np.linspace(0, terrain.width*terrain.horizontal_scale, terrain.width)/10
        y = np.linspace(0, terrain.length*terrain.horizontal_scale, terrain.length)/10   #改变地形粗糙度
        X, Y = np.meshgrid(y, x)
        print(X.shape, Y.shape)
        # terrain.height_field_raw+=Z.astype(terrain.height_field_raw.dtype)
        Z=noise_gen.noise2array(y, x)
        Z = Z.reshape(X.shape) / terrain.vertical_scale  # 放大十倍
        terrain.height_field_raw += Z.astype(terrain.height_field_raw.dtype)
        return