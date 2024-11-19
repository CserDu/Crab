#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

llama2_ckpt_path=/group/40061/cserdu/pretrain/Llama-2-7b-chat-hf
qwen2_ckpt_path=/group/40061/cserdu/pretrain/Qwen2-7B-Instruct

# Training Arguments
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=ms3-s4
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"

python scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path $llama2_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir results/finetune/050-hyper_lora-arig_only/checkpoint-60 \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task True \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task False \
    --adapter_ckpt_path results/finetune/035-hyper_lora-joint_all-arig_1_frame-avs_1_frame/checkpoint-3275/adapter_weights.bin \
    --test_name test_u \
    --device cuda:0 \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /group/40061/cserdu/pretrain/openai-clip-vit-large-patch14-224 \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /group/40061/cserdu/pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 128 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 128 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --output_dir 'test'

