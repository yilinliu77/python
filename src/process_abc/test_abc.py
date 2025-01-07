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
    less10_list = read_list("src/process_abc/abc_less10.txt")
    less30_list = read_list("src/process_abc/abc_less30.txt")
    larger100_list = read_list("src/process_abc/abc_larger100.txt")
    others_list = read_list("src/process_abc/abc_others.txt")
    long_list = read_list("src/process_abc/abc_long_shape.txt")

    exception_list = read_list(r"src/process_abc/exception_list.txt")
    exception_list1 = read_list(r"src/brepnet/data/list/exception1.txt")
    exception_list2 = read_list(r"src/brepnet/data/list/exception2.txt")

    deduplicated_list = read_list(r"src/process_abc/abc_deduplicated_list.txt")
    print("{} samples in deduplicated_list".format(len(deduplicated_list)))

    deduplicated_list = deduplicated_list - less10_list - long_list - larger100_list - others_list - exception_list - exception_list1 - exception_list2
    print("{} samples remains".format(len(deduplicated_list)))

    parsenet_test = read_list(r"src/process_abc/parsenet_test_ids.txt")
    parsenet_valid = read_list(r"src/process_abc/parsenet_val_ids.txt")
    deepcad_valid = read_list(r"src/img2brep/data/deepcad_validation_whole.txt")
    deepcad_test = read_list(r"src/img2brep/data/deepcad_test_whole.txt")

    brepgen_abc_train_test = read_list(r"src/brepnet/data/list/deduplicated_abc_training_brepnet.txt")
    brepgen_abc_validation_test = read_list(r"src/brepnet/data/list/deduplicated_abc_validation_brepnet.txt")
    brepgen_abc_testing_test = read_list(r"src/brepnet/data/list/deduplicated_abc_testing_brepnet.txt")

    ours_validation_list = (brepgen_abc_validation_test & deduplicated_list)
    print("ours_validation_list: ", len(ours_validation_list))
    ours_testing_list = (brepgen_abc_testing_test & deduplicated_list)
    print("ours_testing_list: ", len(ours_testing_list))
    ours_training_list = (brepgen_abc_train_test & deduplicated_list)
    print("ours_training_list: ", len(ours_training_list))
    ours_training_list = ours_training_list - parsenet_test - parsenet_valid - deepcad_valid - deepcad_test
    print("ours_training_list after filtering: ", len(ours_training_list))

    ours_training_list = sorted(list(ours_training_list))
    ours_validation_list = sorted(list(ours_validation_list))
    ours_testing_list = sorted(list(ours_testing_list))

    np.savetxt("src/process_abc/abc_training.txt", ours_training_list, fmt="%s")
    np.savetxt("src/process_abc/abc_validation.txt", ours_validation_list, fmt="%s")
    np.savetxt("src/process_abc/abc_testing.txt", ours_testing_list, fmt="%s")
    np.savetxt("src/process_abc/abc_total.txt", sorted(ours_training_list+ours_validation_list+ours_testing_list), fmt="%s")
