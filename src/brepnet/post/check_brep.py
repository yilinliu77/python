import os, sys
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    dir = Path(sys.argv[1])
    files = [f for f in os.listdir(dir) if os.path.isdir(dir / f)]
    files.sort()
    num_valid = 0
    num_mesh = 0
    num_report = 20
    if (dir / "failed.txt").exists():
        os.remove(dir / "failed.txt")

    for file in tqdm(files):
        if (dir / file / "success.txt").exists():
            num_valid += 1
        else:
            with open(dir / "failed.txt", "a") as f:
                f.write(file + "\n")
            if num_report > 0:
                print(file)
                num_report -= 1
        if (dir / file / "recon_brep.ply").exists():
            num_mesh += 1

    print(f"{num_valid}/{num_mesh}/{len(files)}")
