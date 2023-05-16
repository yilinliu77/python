import os
import numpy as np,cv2
import pyexiv2
import ray
from pyproj import Transformer, CRS
from tqdm import tqdm

img_dir = r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Test_imgs2"
output_dir = r"D:\Projects\NeuralRecon\Test_data\OBL_L7\Colmap"
output_img_dir = os.path.join(output_dir,"imgs")
central_point = np.array([493260.00,2492700.00,0])

@ray.remote
def process_img(img_name):
    img_file = os.path.join(img_dir, img_name)
    img = cv2.imread(img_file, cv2.IMREAD_ANYCOLOR)
    img_meta = pyexiv2.Image(img_file).read_exif()

    latitude = (img_meta['Exif.GPSInfo.GPSLatitude']).split(" ")
    longitude = img_meta['Exif.GPSInfo.GPSLongitude'].split(" ")
    altitude = img_meta['Exif.GPSInfo.GPSAltitude'].split("/")

    latitude = [item.split("/") for item in latitude]
    latitude = [float(item[0]) / float(item[1]) for item in latitude]
    latitude = latitude[0] + latitude[1] / 60.0 + latitude[2] / 3600.0
    longitude = [item.split("/") for item in longitude]
    longitude = [float(item[0]) / float(item[1]) for item in longitude]
    longitude = longitude[0] + longitude[1] / 60.0 + longitude[2] / 3600.0
    altitude = float(altitude[0]) / float(altitude[1])

    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(4547))
    target_vertices = transformer.transform(latitude, longitude, altitude)
    target_vertices = np.asarray([target_vertices[1],target_vertices[0],target_vertices[2]]) - central_point
    cv2.imwrite(os.path.join(output_img_dir, img_name), img)
    return "{},{},{},{}\n".format(img_name,target_vertices[0],target_vertices[1],target_vertices[2])

if __name__ == '__main__':
    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(output_img_dir,exist_ok=True)
    print("Input dir:{}".format(img_dir))
    print("Output dir:{}".format(output_dir))
    print("Central point:{}".format(central_point))
    ray.init(
        # local_mode=True
    )
    tasks = []
    for img_name in os.listdir(img_dir):
        if img_name[-3:] not in ["JPG","jpg","png","PNG"]:
            continue
        tasks.append(process_img.remote(img_name))

    print("Found {} images".format(len(tasks)))
    pose_str = ray.get(tasks)

    with open(os.path.join(output_dir, "pose.txt"), "w") as f:
        for line in pose_str:
            f.write(line)
    print("Done")