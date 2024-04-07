from vehicle_Isaacgym.vehicle.Terrains.Terrain import *

class Slope(Terrain):
    def __init__(self, cfg, num_robots):
        super(Slope, self).__init__(cfg=cfg, num_robots=num_robots)
        num_robots_per_map = int(num_robots / self.env_cols)
        left_over = num_robots % self.env_cols
        for j in range(self.env_cols):
            min_height = 0
            for i in range(self.env_rows):
                terrain = SubTerrain("terrain", width=self.width_per_env_pixels, length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
                difficulty = i / self.env_rows
                choice = j / self.env_cols
                slope = difficulty * 0.1
                if choice < 0.05:
                    slope *= -1
                min_height=sloped_terrain(terrain, slope, min_height)
                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.length_per_env_pixels
                end_y = self.border + (j + 1) * self.length_per_env_pixels
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

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


def sloped_terrain(terrain, slope=1, min_height=0):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)+min_height
    terrain.height_field_raw[:, np.arange(terrain.length)] += ((max_height-min_height) * xx / terrain.width+min_height).astype(terrain.height_field_raw.dtype)
    return max_height