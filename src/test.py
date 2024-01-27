import os
import shutil

root = r"G:/Dataset/GSP/Results/viz_output/0120_random_mesh_imgs"
output_root = r"G:/Dataset/GSP/Results/viz_output/"

if __name__ == '__main__':
    files = list(sorted(set([file[:8] for file in os.listdir(root)])))

    for i in range(len(files)):
        if i % 5 == 0:
            os.makedirs(os.path.join(output_root, str(i//5)), exist_ok=True)

        output_dir = os.path.join(output_root, str(i//5))
        for suffix in ["_complex", "_sed", "_hp", "_gt", "_ours"]:
            shutil.copy(
                os.path.join(root, files[i] + suffix + ".png"),
                os.path.join(output_dir, str(i%5) + suffix + ".png"))
            shutil.copy(
                os.path.join(root, files[i] + suffix + "_curve.png"),
                os.path.join(output_dir, str(i%5) + suffix + "_curve.png"))

    pass