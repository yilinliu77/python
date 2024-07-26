import os, subprocess
from tqdm import tqdm

files = [file for file in os.listdir("/data1/yilin") if file.endswith(".tar")]
for file in tqdm(files):
    subprocess.run(["tar", "-xf", file])
    os.remove("/data1/yilin/" + file)