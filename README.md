# ToRL-Reproduction

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
