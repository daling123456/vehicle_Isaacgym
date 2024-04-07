from .Terrain import Terrain
from .curiculum import Curiculum
from .random_terrain import randomized_terrain
from .slope import Slope

ALL_TERRAINS = {
                    "trimesh": Curiculum,
                    "random": randomized_terrain,
                    "slope": Slope,
                    # "plane": GroundPlaneTerrain
                    }