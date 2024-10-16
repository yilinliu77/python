from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Extend.DataExchange import read_step_file
import os
import argparse
import glob
from tqdm import tqdm

from src.brepnet.post.utils import solid_valid_check


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
    args = parser.parse_args()
    data_root = args.data_root
    folders = os.listdir(data_root)
    step_file_list = load_data_with_prefix(data_root, ".step")

    # Load cad data
    is_valid_list = []
    pbar = tqdm(step_file_list)
    for step_file in pbar:
        solid = read_step_file(step_file)
        analyzer = BRepCheck_Analyzer(solid, True, True, False)
        is_valid = analyzer.IsValid()
        # is_valid = solid_valid_check(solid)
        is_valid_list.append(is_valid)
        pbar.set_postfix({"valid_count": sum(is_valid_list)})

    valid_count = sum(is_valid_list)
    print(f"Number of valid CAD solids: {valid_count}")
