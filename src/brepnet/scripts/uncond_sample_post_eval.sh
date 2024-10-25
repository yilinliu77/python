#!/bin/bash
gt_test_pc_root=$1
ckpt=$2
gaussian_weights=$3
diffusion_type=$4
sample_size=$5
fake_sample_feature_root=$6
fake_post_root=${fake_sample_feature_root}"_post"
fake_post_pcd_root=${fake_post_root}"_pcd"

echo "Sample & Post & Evaluate"
echo "  ckpt: $ckpt"
echo "  diffusion_type: $diffusion_type"
echo "  pad_method: $pad_method"
echo "  sample_size: sample_size"
echo "  gt_test_pc_root: $gt_test_pc_root"
echo "  fake_sample_feature_root: $fake_sample_feature_root"
echo "  fake_post_root: $fake_post_root"
echo "  fake_post_pcd_root: $fake_post_pcd_root"

echo "STep0 Sample"
python -m src.brepnet.train_diffusion model.name=Diffusion_condition model.diffusion_latent=768 trainer.resume_from_checkpoint=${ckpt} trainer.evaluate=true trainer.accelerator=16-mixed trainer.batch_size=1024 model.num_max_faces=30 dataset.num_max_faces=30 trainer.test_output_dir=${fake_sample_feature_root} model.gaussian_weights=${gaussian_weights} model.diffusion_type=${diffusion_type} model.pad_method=random model.sigmoid=false dataset.name=Dummy_dataset dataset.length=${sample_size}

echo "STEP1 Build Brep"
python -m src.brepnet.post.construct_brep --data_root ${fake_sample_feature_root} --out_root ${fake_post_root} --use_ray --use_cuda --num_cpus 16

echo "STEP2 Sample Points"
python -m src.brepnet.eval.sample_points --data_root ${fake_post_root} --out_root ${fake_post_pcd_root} --valid

echo "STEP3 Evaluate"
python -m src.brepnet.eval.eval_brepgen --real ${gt_test_pc_root} --fake ${fake_post_pcd_root}

echo "POST ADN EVAL DONE"
