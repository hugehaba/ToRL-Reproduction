## Results

We report training-time validation accuracy from the ToRL training pipeline. Validation was automatically performed during training using the same sandbox-based Python execution and math-verification scoring setup.

> Note: These are training-time validation results, not a separate post-hoc merged-checkpoint evaluation.

### Validation Accuracy

| Step | AIME24 | MATH500 | AIME25 | Average |
|---:|---:|---:|---:|---:|
| 0 | 10.0% | 48.5% | 3.3% | 20.6% |
| 100 | 13.3% | 51.5% | 10.0% | 24.9% |
| 140 | 16.7% | 57.5% | 16.7% | 30.3% |
| 160 | **30.0%** | 56.7% | 16.7% | **34.5%** |
| 200 | 26.7% | **59.0%** | 13.3% | 33.0% |
| 220 | **30.0%** | 53.7% | **18.3%** | 34.0% |
| 250 | 20.0% | 53.7% | 12.5% | 28.7% |
| 260 | 23.3% | 50.7% | 16.7% | 30.2% |

![ToRL validation accuracy curve](analysis/torl_validation_accuracy_curve.png)

### Key Findings

- MATH500 improved from **48.5%** at step 0 to **59.0%** at step 200.
- AIME24 improved from **10.0%** at step 0 to **30.0%** at step 160/220.
- AIME25 improved from **3.3%** at step 0 to **18.3%** at step 220.
- The best average validation accuracy was **34.5%** at step 160.
- The latest saved checkpoint is `global_step_250`, while the best validation performance appeared earlier. This shows that longer RL training did not monotonically improve all benchmarks.

---

## Environment

The experiment was reproduced on a 2-GPU AutoDL instance.

| Component | Version |
|---|---|
| GPU | 2× NVIDIA H800/A800 80GB |
| Base model | Qwen2.5-Math-7B-Base |
| Training framework | ToRL / verl |
| Training strategy | Full-parameter FSDP + vLLM rollout |
| PyTorch | 2.6.0+cu124 |
| PyTorch CUDA runtime | 12.4 |
| vLLM | 0.8.1 |
| Ray | 2.44.0 |
| Flash Attention | 2.7.4.post1, built from source |
| Transformers | 4.50.0 |
| Precision | bfloat16 |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-Math-7B-Base |
| Algorithm | GRPO |
| Training method | Full-parameter FSDP |
| Rollout engine | vLLM |
| Tensor parallel size | 2 |
| Train batch size | 32 prompts |
| Samples per prompt | 8 |
| Max prompt length | 400 |
| Max response length | 2048 |
| Learning rate | 1e-6 |
| KL coefficient | 0.0 |
| PPO mini batch size | 256 |
| PPO max token length per GPU | 8192 |
| GPU memory utilization | 0.5 |
| Latest saved checkpoint | global_step_250 |
| Final observed step | 260 |

---

## Training Dynamics

The validation curve shows that ToRL improves mathematical reasoning performance during early GRPO training, but performance is not monotonic across all benchmarks.

![Validation curve](analysis/torl_validation_accuracy_curve.png)

The strongest improvements appear between step 100 and step 220. MATH500 peaks at step 200, while AIME24 and AIME25 peak at step 160/220. After step 220, the average validation score drops slightly, suggesting possible instability or over-optimization under the current small-scale 2-GPU setting.# ToRL-Reproduction

Reproduction and training dynamics analysis of **ToRL: Scaling Tool-Integrated RL** ([arXiv 2503.23383](https://arxiv.org/abs/2503.23383)) on 2×NVIDIA A800 80GB.

> **ToRL** trains LLMs to autonomously use Python code execution as a tool via GRPO reinforcement learning, achieving significant improvements on mathematical reasoning benchmarks.

---

## Results

| Checkpoint | MATH500 | AIME 2024 | AIME 2025 |
|-----------|---------|-----------|-----------|
| Base model (step 0) | 50.7% | 10.0% | 10.8% |
| Step 50 | TBD | TBD | TBD |
| Step 100 | TBD | TBD | TBD |
| Final (step 250) | TBD | TBD | TBD |
| ToRL paper (reported) | ~80%+ | 43.3% | — |

*Results will be updated as training completes.*

---

## Environment

| Component | Version |
|-----------|---------|
| Hardware | 2× NVIDIA A800 80GB PCIe |
| PyTorch | 2.6.0 |
| vLLM | 0.8.1 |
| Ray | 2.44.0 |
| Flash Attention | 2.7.4 |
| Transformers | 4.50.0 |
| CUDA | 13.0 |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-Math-7B-Base |
| Training method | Full parameter FSDP (no LoRA) |
| Algorithm | GRPO |
| Rollout batch size | 32 prompts × 8 samples |
| Max response length | 1536 tokens |
| Learning rate | 1e-6 |
| KL coefficient | 0.0 |
| Tool calls per step | 1 (C=1) |
| Total steps | 250 |
| GPU memory utilization | 0.5 |
| Tensor parallel size | 2 |

---

## Key Differences from Paper

| Aspect | Paper | This Reproduction |
|--------|-------|-------------------|
| GPU | 8× A100 (estimated) | 2× A800 |
| Response length | 3072 | 1536 (memory constraint) |
| Total steps | 300 | 250 |
| ppo_max_token_len_per_gpu | 16384 | 8192 |

---

## How to Reproduce

### 1. Clone ToRL and download model

```bash
git clone https://github.com/GAIR-NLP/ToRL
cd ToRL
# Download Qwen2.5-Math-7B-Base to your model directory
```

### 2. Install dependencies

```bash
pip install "qwen-agent[python_executor]"
pip install math-verify
```

### 3. Apply patch

File: `verl/workers/sharding_manager/fsdp_vllm.py`

In the `__enter__` method, add `torch.cuda.empty_cache()` before `wake_up()`:

```python
torch.cuda.empty_cache()
self.inference_engine.wake_up()
```

### 4. Set environment variables

```bash
export VLLM_USE_V1=0
export NCCL_CUMEM_ENABLE=0
```

### 5. Run training

```bash
cd scripts
bash torl_7b_2gpu.sh
```

---

## Critical Bug Fixes

Running ToRL on 2×A800 with vLLM 0.8.1 required solving 11 issues.
See [docs/troubleshooting.md](docs/troubleshooting.md) for full details.

| Issue | Fix |
|-------|-----|
| vLLM V1 engine flashinfer incompatibility | `VLLM_USE_V1=0` |
| NCCL + vLLM cuMem conflict → OOM at wake_up | `NCCL_CUMEM_ENABLE=0` |
| PyTorch cache blocking cuMem remapping | `torch.cuda.empty_cache()` before `wake_up()` |
| `torch_c_dlpack_ext` ABI mismatch | `pip uninstall torch_c_dlpack_ext` |
| Missing Python executor dependencies | `pip install "qwen-agent[python_executor]"` |
| Backward OOM | `ppo_max_token_len_per_gpu=8192` |

---

## Training Dynamics

*Plots will be added after training completes.*

- [ ] Reward / score mean curve
- [ ] Code call rate over training steps
- [ ] Tool execution success rate
- [ ] Response length curve
- [ ] Error type distribution

---

## Project Structure

```
ToRL-Reproduction/
├── scripts/
│   └── torl_7b_2gpu.sh      # Training script (modified)
├── docs/
│   ├── troubleshooting.md   # 11 bugs and fixes
│   └── environment.md       # Full environment setup
├── analysis/                # Training dynamics plots (TBD)
├── results/                 # Eval results (TBD)
└── logs/                    # Training logs (TBD)
```

---

## Citation

```bibtex
@misc{li2025torlscalingtoolintegratedrl,
  title={ToRL: Scaling Tool-Integrated RL},
  author={Xuefeng Li and Haoyang Zou and Pengfei Liu},
  year={2025},
  eprint={2503.23383},
  archivePrefix={arXiv}
}
```

---

## Acknowledgements

Based on the official [GAIR-NLP/ToRL](https://github.com/GAIR-NLP/ToRL) repository.
