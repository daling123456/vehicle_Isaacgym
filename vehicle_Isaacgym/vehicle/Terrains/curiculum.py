from vehicle_Isaacgym.vehicle.Terrains.Terrain import *
class Curiculum(Terrain):
    def __init__(self,cfg, num_robots):
        super().__init__(cfg, num_robots)
        num_robots_per_map = int(num_robots / self.env_cols)
        left_over = num_robots % self.env_cols
        for j in range(self.env_cols):
            for i in range(self.env_rows):
                terrain = SubTerrain("terrain", width=self.width_per_env_pixels, length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale, horizontal_scale=self.horizontal_scale)
                difficulty = i / self.env_rows
                choice = j / self.env_cols
                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                amplitude = 0.1 + difficulty * 0.5
                stepping_stones_size = 2 - 1.8 * difficulty

                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    wave_terrain(terrain, num_waves=1, amplitude=amplitude)
                    # stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                    #                         platform_size=3.)

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
