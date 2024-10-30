# Example: ./src/brepnet/uncond.sh /mnt/d/uncond_results/uncond_gaussian_l2_epsilon_random_1m uncond_gaussian_l2_epsilon_random_1m.ckpt epsilon random 1024 4096
dir=$1
ckpt=$2
diffusion_type=$3
pad_method=$4
batch_size=$5
sample_size=$6

python -m src.brepnet.train_diffusion trainer.resume_from_checkpoint=/mnt/d/uncond_checkpoints/${ckpt} trainer.evaluate=true trainer.accelerator=16-mixed trainer.batch_size=${batch_size} trainer.test_output_dir=${dir} model.name=Diffusion_condition model.diffusion_type=${diffusion_type} model.pad_method=${pad_method} model.gaussian_weights=1e-6 model.sigmoid=false dataset.name=Dummy_dataset dataset.length=${sample_size}

python -m src.brepnet.post.construct_brep --data_root ${dir} --out_root ${dir}_post --use_ray --use_cuda --num_cpus 16