from vehicle_Isaacgym.vehicle.tasks.vehicleTerrain import VehicleTerrain
from vehicle_Isaacgym.vehicle.tasks.vehicle import Vehicle
from vehicle_Isaacgym.vehicle.tasks.Pi_terrain import PiTerrain
from vehicle_Isaacgym.vehicle.tasks.anymal_terrain import AnymalTerrain
from vehicle_Isaacgym.vehicle.tasks.cwego import Cwego


task_map={
    "Cwego": Cwego,
    "Vehicle": Vehicle,
    "VehicleTerrain": VehicleTerrain,
    "PiTerrain": PiTerrain,
    "AnymalTerrain": AnymalTerrain
}