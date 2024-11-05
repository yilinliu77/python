#!/bin/bash
# ./src/brepnet/scripts/uncond_sample_post_eval.sh /mnt/d/deepcad_test_pcd /mnt/d/uncond_checkpoints/uncond_gaussian_epsilon_f30_3m.ckpt 1e-6 epsilon 6000 /mnt/d/uncond_results/uncond_gaussian_epsilon_f30_3m /mnt/d/deepcad_train_v6 ./src/brepnet/data/list/deduplicated_deepcad_training_30.txt

gt_test_pc_root=$1
ckpt=$2
gaussian_weights=$3
diffusion_type=$4
sample_size=$5
fake_sample_feature_root=$6
train_root=$7
train_txt=$8

fake_post_root=${fake_sample_feature_root}"_post"
fake_post_pcd_root=${fake_post_root}"_pcd"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'
variables=(gt_test_pc_root ckpt gaussian_weights diffusion_type sample_size fake_sample_feature_root train_root train_txt fake_post_root fake_post_pcd_root)
for var in "${variables[@]}"; do
    if [ -n "${!var}" ]; then
        echo -e "${GREEN}${var}: ${!var}${NC}"
    else
        echo "${RED}Error: ${var} is empty${NC}"
        exit 1
    fi
done

echo -e "\n${GREEN}STep0 Sample${NC}"
python -m src.brepnet.train_diffusion model.name=Diffusion_condition model.diffusion_latent=768 trainer.resume_from_checkpoint=${ckpt} trainer.evaluate=true trainer.accelerator=16-mixed trainer.batch_size=1024 model.num_max_faces=30 dataset.num_max_faces=30 trainer.test_output_dir=${fake_sample_feature_root} model.gaussian_weights=${gaussian_weights} model.diffusion_type=${diffusion_type} model.pad_method=random model.sigmoid=false dataset.name=Dummy_dataset dataset.length=${sample_size} || exit 1

echo -e "\n${GREEN}STEP1 Build Brep${NC}"
python -m src.brepnet.post.construct_brep --data_root ${fake_sample_feature_root} --out_root ${fake_post_root} --use_ray --use_cuda --num_cpus 40 || exit 1

echo -e "\n${GREEN}STEP2 Sample Points${NC}"
python -m src.brepnet.eval.sample_points --data_root ${fake_post_root} --out_root ${fake_post_pcd_root} --valid || exit 1

echo -e "\n${GREEN}STEP3 Evaluate MMD & COV & JSD ${NC}"
python -m src.brepnet.eval.eval_brepgen --real ${gt_test_pc_root} --fake ${fake_post_pcd_root} || exit 1

echo -e "\n${GREEN}STEP4 Find Nearest in the training set for each sample using CD${NC}"
python -m src.brepnet.viz.find_nearest_pc_cd --fake_post ${fake_post_root} --fake_pcd_root ${fake_post_pcd_root} --train_root ${train_root} --txt ${train_txt}|| exit 1

echo -e "\n${GREEN}STEP5 Evaluate Unique & Novel ${NC}"
python -m src.brepnet.eval.eval_unique_novel --fake_root ${fake_sample_feature_root} --fake_post ${fake_post_root} --train_root ${train_root} --use_ray --txt ${train_txt} --min_face 0 || exit 1

echo -e "\n${GREEN}POST ADN EVAL DONE${NC}"
