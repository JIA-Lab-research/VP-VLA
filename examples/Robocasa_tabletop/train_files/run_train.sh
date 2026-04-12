#!/bin/bash
# VP-VLA Training Script for RoboCasa

# === Distributed training settings ===
# Uncomment and modify if needed for your cluster
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=3600
export TORCH_DISTRIBUTED_DEBUG=DETAIL

###########################################################################################
# === Please modify the following paths according to your environment ===
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list=''
DIT_TYPE="DiT-B"

# Data paths
data_root_dir=./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
visual_prompt_dir=./playground/Datasets/visual_prompt_robocasa_by_frames
extracted_frames_dir=./playground/Datasets/extracted_frames_robocasa
data_mix=fourier_gr1_unified_1000

# Output
run_root_dir=./playground/Checkpoints
run_id=robocasa_visual_prompt_training_${Framework_name}
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla_visual_prompt.py \
  --config_yaml ./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_visual_prompt.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.visual_prompt_dir ${visual_prompt_dir} \
  --datasets.vla_data.extracted_frames_dir ${extracted_frames_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 32 \
  --datasets.vla_data.video_backend decord \
  --datasets.vla_data.feed_both_images true \
  --datasets.vp_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 3e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.loss_scale.visual_prompt 0.1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project robocasa_visual_prompt \
  --wandb_entity your_wandb_entity
