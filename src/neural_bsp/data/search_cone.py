import os
from tqdm import tqdm

files = [file for file in os.listdir(".") if not os.path.isfile(file)]
for file in tqdm(files):
    yml = [yml for yml in os.listdir(file) if yml.endswith(".yml")]
    if len(yml) == 0:
        continue
    yml = yml[0]

    with open(os.path.join(file, yml)) as f:
        lines = f.readlines()
    str = "".join(lines)
    
    if "BSpline" in str or "Cone" not in str:
        continue

    print(file)