
from isaacgym.terrain_utils import *
class Terrain():
    def __init__(self, cfg, num_robots):
        self.horizontal_scale=0.1
        self.border_size = 20
        self.vertical_scale=0.005
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions=[np.sum(cfg["terrainProportions"][:i+1]) for i in range(len(cfg['terrainProportions']))]   #[0.1, 0.2, 0.55, 0.8, 1.0]

        self.cfg=cfg
        self.env_rows=cfg["numLevels"]
        self.env_cols=cfg["numTerrains"]
        self.num_maps= self.env_rows * self.env_cols
        self.env_origins=np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels= int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels= int(self.env_length/ self.horizontal_scale)

        self.border = int(self.border_size/self.horizontal_scale)
        self.total_cols=int(self.env_cols*self.width_per_env_pixels)+2*self.border
        self.total_rows=int(self.env_rows*self.width_per_env_pixels)+2*self.border

        self.height_field_raw=np.zeros((self.total_rows, self.total_cols), dtype=np.int16)
        # if cfg["curriculum"]:
        #     curiculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        # else:
        #     self.randomized_terrain()

    def addheightfield(self):
        self.heightsamples=self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale, self.cfg["slopeTreshold"])

