# construct_brep
python -m src.brepnet.post.construct_brep --data_root E:\data\img2brep\deepcad_whole_train_v5 --out_root E:\data\img2brep\deepcad_whole_train_v5_out

# eval brep
python -m src.brepnet.post.eval_brep --data_root E:\data\img2brep\deepcad_whole_train_v5 --out_root E:\data\img2brep\deepcad_whole_train_v5_out --used_gpu 0 1 2 3 4 5 6 7
