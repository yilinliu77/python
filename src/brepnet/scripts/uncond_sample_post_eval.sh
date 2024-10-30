#!/bin/bash
# ./src/brepnet/scripts/uncond_sample_post_eval.sh /mnt/d/deepcad_test_pcd /mnt/d/uncond_checkpoints/1025_gaussian_epsilon_f730_400k.ckpt 1e-6 epsilon 6000 /mnt/d/uncond_results/1025_gaussian_epsilon_f730_400k

gt_test_pc_root=$1
ckpt=$2
gaussian_weights=$3
diffusion_type=$4
sample_size=$5
fake_sample_feature_root=$6
fake_post_root=${fake_sample_feature_root}"_post"
fake_post_pcd_root=${fake_post_root}"_pcd"

variables=(gt_test_pc_root ckpt gaussian_weights diffusion_type sample_size fake_sample_feature_root fake_post_root fake_post_pcd_root)
for var in "${variables[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is empty"
        exit 1
    fi
done

GREEN='\033[0;32m'
NC='\033[0m'
echo -e "${GREEN}Sample & Post & Evaluate${NC}"
echo -e "${GREEN}gt_test_pc_root: ${gt_test_pc_root}${NC}"
echo -e "${GREEN}ckpt: ${ckpt}${NC}"
echo -e "${GREEN}gaussian_weights: ${gaussian_weights}${NC}"
echo -e "${GREEN}diffusion_type: ${diffusion_type}${NC}"
echo -e "${GREEN}sample_size: ${sample_size}${NC}"
echo -e "${GREEN}fake_sample_feature_root: ${fake_sample_feature_root}${NC}"
echo -e "${GREEN}fake_post_root: ${fake_post_root}${NC}"
echo -e "${GREEN}fake_post_pcd_root: ${fake_post_pcd_root}${NC}"

echo -e "${GREEN}STep0 Sample${NC}"
python -m src.brepnet.train_diffusion model.name=Diffusion_condition model.diffusion_latent=768 trainer.resume_from_checkpoint=${ckpt} trainer.evaluate=true trainer.accelerator=16-mixed trainer.batch_size=1024 model.num_max_faces=30 dataset.num_max_faces=30 trainer.test_output_dir=${fake_sample_feature_root} model.gaussian_weights=${gaussian_weights} model.diffusion_type=${diffusion_type} model.pad_method=random model.sigmoid=false dataset.name=Dummy_dataset dataset.length=${sample_size} || exit 1

echo -e "${GREEN}STEP1 Build Brep${NC}"
python -m src.brepnet.post.construct_brep --data_root ${fake_sample_feature_root} --out_root ${fake_post_root} --use_ray --use_cuda --num_cpus 16 || exit 1

echo -e "${GREEN}STEP2 Sample Points${NC}"
python -m src.brepnet.eval.sample_points --data_root ${fake_post_root} --out_root ${fake_post_pcd_root} --valid || exit 1

echo -e "${GREEN}STEP3 Evaluate MMD & COV & JSD ${NC}"
python -m src.brepnet.eval.eval_brepgen --real ${gt_test_pc_root} --fake ${fake_post_pcd_root} || exit 1

echo -e "${GREEN}STEP3 Evaluate Unique & Novel ${NC}"
echo -e "${GREEN}Find Nearest in the training set for each sample using CD${NC}"
python -m src.brepnet.viz.find_nearest_pc_cd --fake_root /mnt/d/uncond_results/uncond_gaussian_epsilon_7_30_post --fake_pcd_root /mnt/d/uncond_results/uncond_gaussian_epsilon_7_30_post_pcd --train_root /mnt/d/deepcad_train_v6 --txt ./src/brepnet/data/list/deduplicated_deepcad_training_30.txt
echo -e "${GREEN}Computing Unique & Novel${NC}"
python -m src.brepnet.eval.eval_unique_novel --fake /mnt/d/uncond_results/uncond_gaussian_epsilon_7_30 --fake_post /mnt/d/uncond_results/uncond_gaussian_epsilon_7_30_post --train_root /mnt/d/deepcad_train_v6 --use_ray --txt ./src/brepnet/data/list/deduplicated_deepcad_training_30.txt

echo -e "${GREEN}POST ADN EVAL DONE${NC}"
