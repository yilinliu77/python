data_root="/mnt/d/img2brep/0924_0914_dl8_ds256_context_kl_v5_test"
out_root=$data_root"_out"
out_root2=$data_root"_out2"

# construct_brep
python -m src.brepnet.post.move $data_root

# construct_brep
python -m src.brepnet.post.construct_brep --data_root $out_root --out_root $out_root2 --is_cover 1

# eval brep
python -m src.brepnet.post.eval_brep --data_root $out_root2 --out_root $out_root2 --used_gpu 0
