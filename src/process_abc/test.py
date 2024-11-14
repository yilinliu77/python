import numpy as np

def read_list(v_file):
    return set([item.strip() for item in open(v_file).readlines()])

if __name__ == "__main__":
    solid_list = read_list("src/process_abc/abc_single_solid.txt")
    cube_list = read_list("src/process_abc/abc_cube.txt")
    cylinder_list = read_list("src/process_abc/abc_cylinder.txt")
    less7_list = read_list("src/process_abc/abc_less7.txt")
    less30_list = read_list("src/process_abc/abc_less30.txt")
    larger100_list = read_list("src/process_abc/abc_larger100.txt")
    others_list = read_list("src/process_abc/abc_others.txt")
    long_list = read_list("src/process_abc/abc_long_shape.txt")

    print("solid_list: ", len(solid_list))
    print("cube_list: ", len(cube_list))
    print("cylinder_list: ", len(cylinder_list))
    print("less7_list: ", len(less7_list))
    print("less30_list: ", len(less30_list))
    print("larger100_list: ", len(larger100_list))
    print("others_list: ", len(others_list))
    print("long_list: ", len(long_list))

    valid_list = (solid_list - cube_list - cylinder_list - less7_list - larger100_list - others_list - long_list) & less30_list
    print("\nvalid_list: ", len(valid_list))
    valid_list = sorted(list(valid_list))
    np.savetxt("src/process_abc/abc_730.txt", valid_list, fmt="%s")