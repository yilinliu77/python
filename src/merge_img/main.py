import cv2
import os
import ffmpeg
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

root_dir = r"D:\repo\python\outputs\testviz"

def read_img(v_path):
    img = cv2.imread(v_path)
    return img

if __name__ == '__main__':
    file_list = [item for item in os.listdir(root_dir)]
    file_list = list(filter(lambda item:item[:5]=="3d_0_", file_list))
    file_list = sorted(file_list, key=lambda item: int(item[:-4].split("_")[2]))
    shape = cv2.imread(os.path.join(root_dir, file_list[0])).shape
    size = (shape[1],shape[0])
    # out = cv2.VideoWriter(os.path.join(root_dir, "project.mp4"), cv2.VideoWriter_fourcc(*'avc1'), 15, size)
    out = cv2.VideoWriter(os.path.join(root_dir, "project.avi"), 0, 15, size)

    print("1. Start to read img files")
    imgs = thread_map(read_img, [os.path.join(root_dir, item) for item in file_list])

    print("2. Start to export video")
    for img in tqdm(imgs):
        out.write(img)
    out.release()

    print("3. Start to encode")
    stream = ffmpeg.input(os.path.join(root_dir, "project.avi"))
    stream = ffmpeg.output(
        stream,
        os.path.join(root_dir, "project.mp4"),
        vcodec='h264',
        pix_fmt='nv21',
        **{'b:v': 20000000},
    )
    ffmpeg.run(stream, overwrite_output=True)