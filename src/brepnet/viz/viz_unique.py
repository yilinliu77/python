import os
import tqdm
import argparse
import ast

from src.brepnet.viz.sort_and_merge import arrange_meshes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Construct Brep From Data')
    parser.add_argument('--fake_post', type=str, required=True)
    parser.add_argument('--out_root', type=str, required=True)
    parser.add_argument('--txt', type=str, required=True)
    args = parser.parse_args()
    fake_post = args.fake_post
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)

    if not os.path.exists(args.txt):
        raise ValueError(f"File not found: {args.txt}")

    with open(args.txt) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    all_components_string = lines[1:]
    all_components_string.sort(key=len, reverse=True)
    for component_idx, component_string in enumerate(all_components_string):
        # "Component: ['00000000', '00006259', '00004750', '00004216', '00007682', '00000861', '00006491', '00002490', '00001832',
        # '00000313', '00003810', '00000140', '00001905']"
        component_string = component_string.replace("Component: ", "")
        component = ast.literal_eval(component_string)
        component = [os.path.join(fake_post, folder, "recon_brep.stl") for folder in component if
                     os.path.exists(os.path.join(fake_post, folder, "recon_brep.stl"))]
        if len(component) == 0:
            print(f"Component {component_idx} is empty.")
            continue
        component_arranged_mesh_path = os.path.join(out_root, f"component_{component_idx}.ply")
        arrange_meshes(component, component_arranged_mesh_path)
        print(f"Component {component_idx} is saved to {component_arranged_mesh_path}.")
    print("All components done.")
