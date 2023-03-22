import cv2
import os

from tqdm import tqdm

root_dir = r"D:\repo\python\output\img_field_test\imgs_log_bb"

if __name__ == '__main__':
    file_list = [item for item in os.listdir(root_dir)]
    file_list = list(filter(lambda item:item[:5]=="3d_3_", file_list))
    file_list = sorted(file_list, key=lambda item: int(item[:-4].split("_")[2]))
    shape = cv2.imread(os.path.join(root_dir, file_list[0])).shape
    size = (shape[1],shape[0])
    out = cv2.VideoWriter(os.path.join(root_dir, "project.mp4"), cv2.VideoWriter_fourcc(*'avc1'), 15, size)

    for i in tqdm(range(len(file_list))):
        img = cv2.imread(os.path.join(root_dir, file_list[i]))
        out.write(img)
    out.release()