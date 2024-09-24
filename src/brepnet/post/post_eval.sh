gt_root="E:\data\img2brep\deepcad_whole_v5\deepcad_whole_test_v5"
data_root="E:\data\img2brep\0924_0914_dl8_ds256_context_kl_v5_test"
out_root=$data_root"_out"
#out_root2=$data_root"_out2"

# construct_brep
# python -m src.brepnet.post.move $data_root

# construct_brep
python -m src.brepnet.post.construct_brep --data_root $data_root --out_root $out_root

# eval brep
python -m src.brepnet.post.eval_brep --eval_root $out_root --gt_root $gt_root
