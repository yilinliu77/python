import airsim
import numpy as np
from airsim import DrivetrainType, YawMode
from tqdm import tqdm

player_start = np.array([32710.0, 20650.0, 7980.0])
way_points = np.array([
    [32710.0, 7130.0, 8840.0],
    [17780.0, 10420.0, 7040.0],
    [-3430.0, 15140.0, 14480.0],
    [-1070.0, 18720.0, 11770.0],
    [960.0, 28250.0, 7850.0]
])

for id_item,_ in enumerate(way_points):
    way_points[id_item] = (way_points[id_item] - player_start) / 100
    way_points[id_item][2] = -way_points[id_item][2]

if __name__ == '__main__':
    client = airsim.MultirotorClient()
    client.confirmConnection()

    objs = client.simListSceneObjects()
    used_color = 1
    for item in tqdm(objs):
        if "Tile_" == item[:5]:
            client.simSetSegmentationObjectID(item, used_color)
            used_color += 1
            if used_color > 255:
                used_color = 1
        else:
            client.simSetSegmentationObjectID(item, 0)

    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()

    for item in way_points:
        client.moveToPositionAsync(
            item[0], item[1], item[2],
            10,
            30,
            drivetrain=DrivetrainType.ForwardOnly,
            yaw_mode=YawMode(False)
        ).join()
