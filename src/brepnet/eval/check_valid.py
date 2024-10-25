import shutil

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.TopAbs import TopAbs_SOLID
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.ShapeFix import ShapeFix_ShapeTolerance
import os
import argparse
import glob
from tqdm import tqdm

from src.brepnet.post.utils import solid_valid_check


def check_step_valid_soild(step_file, precision=1e-1):
    try:
        shape = read_step_file(step_file, as_compound=False, verbosity=False)
    except:
        return False
    if shape.ShapeType() != TopAbs_SOLID:
        return False
    shape_tol_setter = ShapeFix_ShapeTolerance()
    shape_tol_setter.SetTolerance(shape, precision)
    analyzer = BRepCheck_Analyzer(shape)
    is_valid = analyzer.IsValid()
    return is_valid


def load_data_with_prefix(root_folder, prefix):
    data_files = []

    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prefix", type=str, required=False, default="")
    args = parser.parse_args()
    data_root = args.data_root
    folders = os.listdir(data_root)

    if args.prefix:
        step_file_list = load_data_with_prefix(os.path.join(data_root, args.prefix), ".step")
        assert len(step_file_list) > 0
        print(f"Checking CAD solids in {args.prefix}...")
        isvalid = check_step_valid_soild(step_file_list[0])
        print("Valid" if isvalid else "Invalid")
        exit(0)

    step_file_list = load_data_with_prefix(data_root, ".step")

    print(f"Total sample features: {len(folders)}")
    print(f"Total CAD solids: {len(step_file_list)}")

    print("Start checking CAD solids...")

    exception_folders = []
    exception_out_root = data_root + "_exception"
    if os.path.exists(exception_out_root):
        shutil.rmtree(exception_out_root)
    os.makedirs(exception_out_root, exist_ok=False)

    # Load cad data
    valid_count = 0
    pbar = tqdm(step_file_list)
    for step_file in pbar:
        is_valid = check_step_valid_soild(step_file)
        if os.path.exists(os.path.join(os.path.dirname(step_file), "success.txt")) and not is_valid:
            folder_name = os.path.basename(os.path.dirname(step_file))
            exception_folders.append(folder_name)
            shutil.copytree(os.path.dirname(step_file), os.path.join(exception_out_root, folder_name))

        if is_valid:
            valid_count += 1
        pbar.set_postfix({"valid_count": valid_count})

    print(f"Number of valid CAD solids: {valid_count}")
    print(f"Valid rate: {valid_count / len(folders) * 100:.2f}%")

    if len(exception_folders) > 0:
        with open(os.path.join(exception_out_root, "exception_folders.txt"), "w") as f:
            for folder in exception_folders:
                f.write(folder + "\n")
        print(f"Exception folders are saved to {exception_out_root}")
    if len(exception_folders) == 0:
        shutil.rmtree(exception_out_root)
        print("No exception folders found.")
