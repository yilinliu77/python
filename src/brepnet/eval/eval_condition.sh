#!/bin/bash
# sh src/brepnet/eval/eval_condition.sh E:\\data\\img2brep\\ourgen\\deepcad_v6\\deepcad_test_v6 E:\\data\\img2brep\\ourgen\\test\\mv_gaussian_600k_post E:\\data\\img2brep\\ourgen\\test\\mv_pure_600k_post E:\\data\\img2brep\\ourgen\\test\\pc_gaussian_500k_post E:\\data\\img2brep\\ourgen\\test\\pc_pure_500k_post E:\\data\\img2brep\\ourgen\\test\\sv_pure_600k_post

gt_root=$1
shift

for eval_root in "$@"
do
    python -m src.brepnet.eval.eval_condition --gt_root "$gt_root" --eval_root "$eval_root" --use_ray
done