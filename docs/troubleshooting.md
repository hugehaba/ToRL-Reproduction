# Troubleshooting: ToRL on 2×A800 with vLLM 0.8.1

Running ToRL on AutoDL 2×A800 (PyTorch 2.6.0, vLLM 0.8.1) required fixing 11 issues.
Documented here for anyone reproducing in a similar environment.

---

## Issue 1: torch_c_dlpack_ext ABI Mismatch

**Error:**
```
OSError: libtorch_c_dlpack_addon_torch26-cuda.so: undefined symbol:
_ZNK3c106Device3strB5cxx11Ev
```

**Cause:** `torch_c_dlpack_ext` was compiled against a different libtorch C++ ABI than PyTorch 2.6.0.

**Fix:**
```bash
pip uninstall torch_c_dlpack_ext -y
```

verl falls back to Ray Object Store serialization. No impact on training correctness.

---

## Issue 2: vLLM V1 Engine flashinfer Incompatibility

**Error:** vLLM V1 engine attempts to use flashinfer which is not installed.

**Fix:**
```bash
export VLLM_USE_V1=0
```

Add to training script header. vLLM falls back to V0 engine which works correctly.

---

## Issue 3: Missing qwen-agent python_executor Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'pebble'
ImportError: The dependencies for Python Executor support are not installed.
Please install: pip install "qwen-agent[python_executor]"
```

**Cause:** `qwen-agent` base install does not include code execution sandbox dependencies.

**Fix:**
```bash
pip install "qwen-agent[python_executor]"
```

---

## Issue 4: vLLM wake_up OOM — NCCL and vLLM cuMem Conflict (Core Issue)

**Error:**
```
RuntimeError: CUDA Error: out of memory at cumem_allocator.cpp:62
```

**Cause:** Both NCCL and vLLM use the CUDA virtual memory (cuMem) allocator.
vLLM uses cuMem for its sleep/wake KV cache mechanism.
NCCL (newer versions) also uses cuMem for internal communication buffers.
They compete for physical memory pages in the cuMem pool.
When vLLM calls `wake_up()` → `cuMemMap()`, NCCL has consumed physical pages,
leaving insufficient memory for remapping.

**Fix 1: Disable NCCL cuMem**
```bash
export NCCL_CUMEM_ENABLE=0
```
Forces NCCL to use standard `cudaMalloc` instead of cuMem, leaving the cuMem pool for vLLM exclusively.

**Fix 2: Release PyTorch cache before wake_up**

File: `verl/workers/sharding_manager/fsdp_vllm.py`

```python
# In __enter__ method, before wake_up():
torch.cuda.empty_cache()
self.inference_engine.wake_up()
```

Forces PyTorch's caching allocator to return cached pages to the driver
before vLLM attempts cuMemMap remapping.

---

## Issue 5: Residual Processes Occupying GPU Memory

**Error:**
```
GPU 0 has 79.25 GiB total, only 20.31 MiB is free.
Process 74057 has 78.39 GiB memory in use.
```

**Cause:** Previous crashed processes did not release GPU memory.

**Fix:** Always clean up before restarting:
```bash
pkill -9 -f python && pkill -9 -f ray && sleep 3
nvidia-smi  # Verify 0MiB usage before starting
```

---

## Issue 6: Weight Sync OOM with gpu_memory_utilization=0.6

**Error:**
```
torch.OutOfMemoryError: Tried to allocate 1.02 GiB.
GPU 0 only has 831.94 MiB free.
```

**Location:** `dtensor_weight_loaders.py` (FSDP to vLLM weight synchronization)

**Cause:** vLLM KV cache (38.87 GiB) + FSDP + NCCL buffers left no room for
weight sync temporary buffers.

**Fix:** Reduce `gpu_memory_utilization` from 0.6 to 0.5:
```
actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

KV cache reduces from 38.87 GiB to 30.94 GiB, freeing ~8 GiB for weight sync.
No impact on training results.

---

## Issue 7: expandable_segments Incompatible with cuMem

**Error:**
```
AssertionError: Expandable segments are not compatible with memory pool.
Please track https://github.com/pytorch/pytorch/issues/147851
```

**Cause:** vLLM source code explicitly forbids `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
when using the cuMem allocator.

**Fix:** Do not set `expandable_segments:True`. Reducing `gpu_memory_utilization` to 0.5 is sufficient.

---

## Issue 8: Missing math_verify Package

**Error:**
```
ModuleNotFoundError: No module named 'math_verify'
```

**Location:** Validation phase reward computation.

**Fix:**
```bash
pip install math-verify
```

---

## Issue 9: samples_save_path=None TypeError

**Error:**
```
TypeError: expected str, bytes or os.PathLike object, not NoneType
os.path.join(self.config.trainer.samples_save_path, split)
```

**Cause:** Training script did not set `samples_save_path` but code attempts to write to it.

**Fix:** Add to training script:
```
trainer.samples_save_path=/root/autodl-tmp/output_torl/samples
```

---

## Issue 10: Backward OOM with Default ppo_max_token_len_per_gpu

**Error:**
```
torch.OutOfMemoryError: Tried to allocate 4.50 GiB.
GPU 0 only has 2.78 GiB free.
```

**Location:** `dp_actor.py` → `loss.backward()`

**Cause:** Default `ppo_max_token_len_per_gpu=16384` causes peak activation memory
during backward to exceed available GPU memory.

**Fix:** Halve the value:
```
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
```

Equivalent to gradient accumulation — no impact on training results.

---

## Issue 11: NCCL DistBackendError at barrier() After Multiple Crashes

**Error:**
```
torch.distributed.DistBackendError: NCCL error
ncclUnhandledCudaError: Cuda failure 2 'out of memory'
```

**Location:** `fsdp_workers.py` → `_build_model_optimizer()` → `torch.distributed.barrier()`

**Cause:** Accumulated residual GPU state from multiple previous crashes.

**Fix:** Hard cleanup before every restart:
```bash
pkill -9 -f python && pkill -9 -f ray && sleep 3
nvidia-smi  # Must show 0MiB before proceeding
```

---

## Summary

| # | Issue | Fix |
|---|-------|-----|
| 1 | torch_c_dlpack_ext ABI mismatch | `pip uninstall torch_c_dlpack_ext` |
| 2 | vLLM V1 flashinfer error | `VLLM_USE_V1=0` |
| 3 | Missing pebble/python_executor | `pip install "qwen-agent[python_executor]"` |
| 4 | NCCL+vLLM cuMem OOM at wake_up | `NCCL_CUMEM_ENABLE=0` + `empty_cache` patch |
| 5 | Residual processes holding GPU | `pkill` + `nvidia-smi` verify |
| 6 | Weight sync OOM | `gpu_memory_utilization=0.5` |
| 7 | expandable_segments conflict | Remove that env var |
| 8 | math_verify missing | `pip install math-verify` |
| 9 | samples_save_path None | Add path to training script |
| 10 | Backward OOM | `ppo_max_token_len_per_gpu=8192` |
| 11 | NCCL barrier OOM after crashes | Hard cleanup before restart |

## Final Working Environment Variables

```bash
export VLLM_USE_V1=0           # Disable V1 engine
export NCCL_CUMEM_ENABLE=0     # NCCL uses cudaMalloc instead of cuMem
```

## Final Working Training Parameters

| Parameter | Default | This Reproduction |
|-----------|---------|-------------------|
| gpu_memory_utilization | 0.6 | 0.5 |
| ppo_max_token_len_per_gpu | 16384 | 8192 |
| max_response_length | 3072 | 1536 |
| total_epochs | 300 | 250 |
