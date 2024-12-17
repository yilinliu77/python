from random import shuffle

import numpy as np
from lightning_fabric import seed_everything


def read_list(v_file):
    return set([item.strip() for item in open(v_file).readlines()])

if __name__ == "__main__":
    seed_everything(0)
    solid_list = read_list("src/process_abc/abc_single_solid.txt")
    cube_list = read_list("src/process_abc/abc_cube.txt")
    cylinder_list = read_list("src/process_abc/abc_cylinder.txt")
    less7_list = read_list("src/process_abc/abc_less7.txt")
    less30_list = read_list("src/process_abc/abc_less30.txt")
    larger100_list = read_list("src/process_abc/abc_larger100.txt")
    others_list = read_list("src/process_abc/abc_others.txt")
    long_list = read_list("src/process_abc/abc_long_shape.txt")

    exception_list = read_list(r"src/process_abc/exception_list.txt")
    exception_list1 = read_list(r"src/brepnet/data/list/exception1.txt")
    exception_list2 = read_list(r"src/brepnet/data/list/exception2.txt")

    deduplicated_list = read_list(r"src/process_abc/abc_deduplicated_list.txt")
    print("{} samples in deduplicated_list".format(len(deduplicated_list)))

    deduplicated_list = deduplicated_list - long_list - larger100_list - others_list
    print("{} samples remains".format(len(deduplicated_list), len(deduplicated_list) + len(others_list) + len(long_list) + len(larger100_list)))

    parsenet_test = read_list(r"src/process_abc/parsenet_test_ids.txt")
    parsenet_valid = read_list(r"src/process_abc/parsenet_val_ids.txt")
    deepcad_valid = read_list(r"src/img2brep/data/deepcad_validation_whole.txt")
    deepcad_test = read_list(r"src/img2brep/data/deepcad_test_whole.txt")

    valid_test_list = list(deduplicated_list - parsenet_test - parsenet_valid - deepcad_valid - deepcad_test)
    shuffle(valid_test_list)
    valid_list = (set(deepcad_valid) | set(parsenet_valid)) & deduplicated_list
    num_remain = 20000 - len(valid_list)
    valid_list = valid_list | set(valid_test_list[:num_remain])
    print("valid_list: ", len(valid_list))
    valid_test_list = valid_test_list[num_remain:]

    test_list = (set(deepcad_test) | set(parsenet_test)) & deduplicated_list
    num_remain = 20000 - len(test_list)
    test_list = test_list | set(valid_test_list[:num_remain])
    print("test_list: ", len(test_list))

    training_list = deduplicated_list - valid_list - test_list
    print("training_list: ", len(training_list))

    training_list = sorted(list(training_list))
    valid_list = sorted(list(valid_list))
    test_list = sorted(list(test_list))

    # valid_list = (solid_list - cube_list - cylinder_list - less7_list - larger100_list - others_list - long_list) & less30_list
    valid_list = solid_list - cube_list - cylinder_list - less7_list - larger100_list
    print("\nvalid_list: ", len(valid_list))
    valid_list = sorted(list(valid_list))
    np.savetxt("src/process_abc/abc_complete.txt", valid_list, fmt="%s")