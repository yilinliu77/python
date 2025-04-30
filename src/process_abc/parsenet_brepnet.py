from src.process_abc.test import read_list

total_list = [item.strip()[:8] for item in open(r"src/process_abc/parsenet_train_ids.txt").readlines()]

solid_list = read_list("src/process_abc/abc_single_solid.txt")
cube_list = read_list("src/process_abc/abc_cube.txt")
cylinder_list = read_list("src/process_abc/abc_cylinder.txt")
less7_list = read_list("src/process_abc/abc_less7.txt")
less30_list = read_list("src/process_abc/abc_less30.txt")
larger100_list = read_list("src/process_abc/abc_larger100.txt")
others_list = read_list("src/process_abc/abc_others.txt")
long_list = read_list("src/process_abc/abc_long_shape.txt")

print("Ori: ", len(total_list))
print("Only solid: ", len(set(total_list) & set(solid_list)))
print("Without cube: ", len(set(total_list) & set(solid_list) - set(cube_list)))
print("Without cylinder: ", len(set(total_list) & set(solid_list) - set(cube_list) - set(cylinder_list)))
print("Without less7: ", len(set(total_list) & set(solid_list) - set(cube_list) - set(cylinder_list) - set(less7_list)))
print("With less30: ", len((set(total_list) & set(solid_list) - set(cube_list) - set(cylinder_list) - set(less7_list)) & set(less30_list)))
print("Without others: ", len((set(total_list) & set(solid_list) - set(cube_list) - set(cylinder_list) - set(less7_list) & set(less30_list)) - set(others_list)))
print("Without long: ", len((set(total_list) & set(solid_list) - set(cube_list) - set(cylinder_list) - set(less7_list) & set(less30_list)) - set(others_list) - set(long_list)))