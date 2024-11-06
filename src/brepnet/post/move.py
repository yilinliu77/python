import os, shutil
from tqdm import tqdm
import argparse, sys

from src.brepnet.eval.check_valid import check_step_valid_soild
from src.brepnet.post.utils import *

from OCC.Extend.DataExchange import read_step_file, write_step_file, write_stl_file

from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

data_root = r"E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test_out"


def move(data_root):
    save_root = data_root + "_out"
    # save_root = r"/mnt/d/img2brep/0909_test_out"
    os.makedirs(save_root, exist_ok=True)
    all_files = os.listdir(data_root)
    all_files.sort()

    folder_names = []
    for filename in tqdm(all_files):
        if filename.endswith("_feature.npz"):
            folder_names = filename.split("_")[0]
            tgt_path = os.path.join(save_root, folder_names)
            os.makedirs(tgt_path, exist_ok=True)
            shutil.copy(os.path.join(data_root, folder_names + ".npz"), os.path.join(tgt_path, "data.npz"))


def count_success(data_root):
    failed_root = data_root + "_failed"
    os.makedirs(failed_root, exist_ok=True)
    all_files = os.listdir(data_root)
    all_files.sort()
    failed_folder = []

    for filename in tqdm(all_files):
        if os.path.exists(os.path.join(data_root, filename, "recon_brep.step")):
            continue
        else:
            failed_folder.append(filename)
            pass
            # shutil.copytree(os.path.join(data_root, filename), os.path.join(failed_root, filename))

    print(f"Total {len(failed_folder)} failed folders")
    print(f"succeed rate {(len(all_files) - len(failed_folder)) / len(all_files)}")


def check_eval(data_root):
    all_files = os.listdir(data_root)
    all_files.sort()

    for filename in tqdm(all_files):
        if os.path.exists(os.path.join(data_root, filename, "eval.npz")):
            continue
        else:
            print(filename)


def calculate_valid_percentage(shape):
    total_faces = 0
    valid_faces = 0

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = explorer.Current()
        total_faces += 1
        face_analyzer = BRepCheck_Analyzer(face)
        if face_analyzer.IsValid():
            valid_faces += 1
        explorer.Next()

    if total_faces > 0:
        valid_percentage = (valid_faces / total_faces) * 100
        return valid_percentage
    return 0


def check_invalid(data_root):
    all_files = [item for item in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, item))]
    all_files.sort()

    exception_save_root = data_root + "_exception"
    failed_save_root = data_root + "_failed"
    os.makedirs(exception_save_root, exist_ok=True)

    exception_folder = []
    exception_folder_validity_percentage = []
    failed_folder = []

    for filename in tqdm(all_files):
        if os.path.exists(os.path.join(data_root, filename, "recon_brep.step")):
            try:
                is_valid = check_step_valid_soild(os.path.join(data_root, filename, "recon_brep.step"))
                if not is_valid:
                    gen_shape = read_step_file(os.path.join(data_root, filename, "recon_brep.step"), verbosity=False)
                    validity_percentage = calculate_valid_percentage(gen_shape)
                    exception_folder_validity_percentage.append(validity_percentage)
                    exception_folder.append(filename)
                    if os.path.exists(os.path.join(exception_save_root, filename)):
                        continue
                    shutil.copytree(os.path.join(data_root, filename), os.path.join(exception_save_root, filename))
            except:
                print(f"Error in {filename}")
                exception_folder.append(filename)
        else:
            failed_folder.append(filename)
            if os.path.exists(os.path.join(failed_save_root, filename)):
                continue
            shutil.copytree(os.path.join(data_root, filename), os.path.join(failed_save_root, filename))

    print(f"Total {len(exception_folder)} invalid folders")
    print(exception_folder)
    print(exception_folder_validity_percentage)
    with open(os.path.join(exception_save_root, "exception.txt"), "w") as f:
        for except_folder, val_per in zip(exception_folder, exception_folder_validity_percentage):
            f.write(f"{except_folder} {val_per}\n")
        f.write(f"Average validity percentage: {sum(exception_folder_validity_percentage) / len(exception_folder_validity_percentage)}")
    print(f"Average validity percentage: {sum(exception_folder_validity_percentage) / len(exception_folder_validity_percentage)}")


def seg_by_face_num(data_root):
    seg_root = data_root.rstrip('\\') + '_seg_by_face_num'
    if os.path.exists(seg_root):
        shutil.rmtree(seg_root)
    os.makedirs(seg_root, exist_ok=False)
    seg_save_root = [os.path.join(seg_root, 'face_30'),
                     os.path.join(seg_root, 'face_20'),
                     os.path.join(seg_root, 'face_10'),
                     os.path.join(seg_root, 'face_0'),
                     os.path.join(seg_root, 'else')]
    for each in seg_save_root:
        os.makedirs(each, exist_ok=True)

    all_folders = [folder for folder in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, folder))]
    for filename in tqdm(all_folders):
        if os.path.exists(os.path.join(data_root, filename, "recon_brep.step")):
            try:
                gen_shape = read_step_file(os.path.join(data_root, filename, "recon_brep.step"), verbosity=False)
                # get the face num
                explorer = TopExp_Explorer(gen_shape, TopAbs_FACE)
                face_num = 0
                while explorer.More():
                    face_num += 1
                    explorer.Next()
                if face_num > 30:
                    shutil.copytree(os.path.join(data_root, filename), os.path.join(seg_save_root[0], filename))
                elif face_num > 20:
                    shutil.copytree(os.path.join(data_root, filename), os.path.join(seg_save_root[1], filename))
                elif face_num > 10:
                    shutil.copytree(os.path.join(data_root, filename), os.path.join(seg_save_root[2], filename))
                else:
                    shutil.copytree(os.path.join(data_root, filename), os.path.join(seg_save_root[3], filename))
            except:
                print(f"Error in {filename}")
                continue

    print(f"Seg by face num, saved in : {seg_root}")


if __name__ == '__main__':
    if sys.argv[1] == "move":
        move(sys.argv[2])
    elif sys.argv[1] == "seg":
        seg_by_face_num(sys.argv[2])
    else:
        check_invalid(sys.argv[2])
    # count_success(data_root)
    # check_eval(data_root)
