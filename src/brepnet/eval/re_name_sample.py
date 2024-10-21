import os
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", type=str, required=True)
    parser.add_argument("--gen_root", type=str, required=True)
    args = parser.parse_args()

    gt_files = [f for f in os.listdir(args.gt_root) if os.path.isdir(os.path.join(args.gt_root, f))]
    gen_files = [f for f in os.listdir(args.gen_root) if os.path.isdir(os.path.join(args.gen_root, f))]

    assert len(gt_files) == len(gen_files), "The number of files in the ground truth and generated folders should be the same."

    for i in tqdm(range(len(gt_files))):
        os.rename(os.path.join(args.gen_root, gen_files[i]), os.path.join(args.gen_root, gt_files[i] + "_gen"))

    gen_files = [f for f in os.listdir(args.gen_root) if os.path.isdir(os.path.join(args.gen_root, f))]
    for i in tqdm(range(len(gt_files))):
        os.rename(os.path.join(args.gen_root, gt_files[i] + '_gen'), os.path.join(args.gen_root, gt_files[i]))
