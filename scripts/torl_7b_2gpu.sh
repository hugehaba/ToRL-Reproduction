#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/ToRL:$PYTHONPATH
export VLLM_USE_V1=0
export NCCL_CUMEM_ENABLE=0

policy_path=/root/autodl-tmp/models/Qwen2.5-Math-7B-Base
rollout_batch_size=32
n_samples_per_prompts=8
episode=300
temperature=1.0
batch_size=256
lr=1e-6
kl_loss_coef=0.0
kl_coef=0.0
entropy_coeff=0
max_gen_length=3072
numcall=1
run_name=torl.7b.2gpu
data_path=/root/autodl-tmp/ToRL/data/torl_data
save_path=/root/autodl-tmp/output_torl/${run_name}

mkdir -p $save_path

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/test.parquet \
    data.train_batch_size=$rollout_batch_size \
    data.val_batch_size=512 \
    data.max_prompt_length=400 \
    data.max_response_length=$max_gen_length \
    data.template_type=tir_base_0309 \
    actor_rollout_ref.model.path=$policy_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.num_llm_calls_available=$numcall \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_samples_per_prompts \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_seqs=256 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.logger=['console'] \
    trainer.project_name=torl \
    trainer.experiment_name=${run_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$save_path \
    trainer.resume_mode=auto \
    trainer.samples_save_path=/root/autodl-tmp/output_torl/samples \
    trainer.total_epochs=$episode \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
