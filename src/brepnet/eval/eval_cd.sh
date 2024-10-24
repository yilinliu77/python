#!/bin/bash
#gt_test_pc_root = "E:\data\img2brep\deepcad_test_pcd"
#fake_post_root = "E:\data\img2brep\v3\epsilon_random_f64_2m_post2"
#fake_post_sample_pc_root = fake_post_root + "_spc"

# ./src/brepnet/eval/eval_cd.sh E:\\data\\img2brep\\deepcad_test_pcd E:\\data\\img2brep\\brepgen_samples_deepcad
# ./src/brepnet/eval/eval_cd.sh E:\\data\\img2brep\\deepcad_test_pcd E:\\data\\img2brep\\v3\\epsilon_random_f64_2m_post2

gt_test_pc_root=$1
fake_post_root=$2
fake_post_sample_pc_root=${fake_post_root}"_spc"

echo "gt_test_pc_root: $gt_test_pc_root"
echo "fake_post_root: $fake_post_root"
echo "fake_post_sample_pc_root: $fake_post_sample_pc_root"

# sample points
python -m src.brepnet.eval.sample_points --data_root ${fake_post_root} --out_root ${fake_post_sample_pc_root} --valid

# evaluate
python -m src.brepnet.eval.eval_brepgen --fake ${fake_post_sample_pc_root} --real ${gt_test_pc_root}



