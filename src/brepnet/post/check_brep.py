import os, sys
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":
    dir = Path(sys.argv[1])
    files = os.listdir(dir)
    files.sort()
    num_valid = 0
    num_mesh = 0
    num_report = 20
    for file in tqdm(files):
        if (dir/file/"success.txt").exists():
            num_valid+=1
        else:
            if num_report>0:
                print(file)
                num_report-=1
        if (dir/file/"recon_brep.ply").exists():
            num_mesh+=1

    print(f"{num_valid}/{num_mesh}/{len(files)}")
