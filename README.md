# Omni-Attention: Poly-Accelerator AI Infrastructure

**A Unified High-Performance Compute Framework for Heterogeneous AI Infrastructure**

Omni-Attention is a comprehensive engineering project demonstrating mastery over the modern AI infrastructure stack. As foundation models scale beyond the trillion-parameter mark, building and optimizing infrastructure has evolved into a critical strategic asset. This repository transitions standard development into state-of-the-art systems architecture by building the low-level primitives that enable execution at scale across divergent hardware.

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Architectural Divergence](#2-architectural-divergence-hopper-vs-tpu-v5e)
3. [Core Features & Modules](#3-core-features--modules)
4. [Repository Structure](#4-repository-structure)
5. [Installation & Setup](#5-installation--setup)
6. [Usage & Testing](#6-usage--testing)
7. [Omni-Serve C++ Infrastructure](#7-omni-serve-c-infrastructure)
8. [Profiling & Verification](#8-profiling--verification)
9. [References](#9-references)

---

## 1. Executive Summary
The contemporary landscape of Artificial Intelligence infrastructure is defined by a singular, relentless economic pressure: the cost of compute. Omni-Attention tackles this by building a unified, distributed attention mechanism that operates natively on both NVIDIA Hopper GPUs (via OpenAI Triton) and Google TPU v5e/Trillium architectures (via JAX Pallas). 

Crucially, to satisfy the rigorous infrastructure requirements of modern cloud deployments, this framework includes **Omni-Serve**, a custom C++ inference runtime that orchestrates these kernels without the overhead of the Python Global Interpreter Lock (GIL).

---

## 2. Architectural Divergence: Hopper vs. TPU v5e
Modern AI accelerators target two distinct architectural philosophies. A unified framework must demonstrate fluency across both.

| Feature | NVIDIA H100 (GPU) | Google TPU v5e (TPU) |
| :--- | :--- | :--- |
| **Compute Core** | Streaming Multiprocessor (SM)  | Matrix Multiply Unit (MXU) + VPU  |
| **Parallelism Model** | SIMT (Threads/Warps)  | SIMD / Systolic Dataflow  |
| **Memory Management** | HBM → L2 → Shared Mem (Hardware Cache + TMA)  | HBM → VMEM (Explicit DMA)  |
| **Programming Model** | CUDA/PTX/Triton  | XLA HLO / Mosaic / Pallas  |
| **Interconnect** | NVLink/NVSwitch  | ICI (Inter-Chip Interconnect) / Torus  |

---

## 3. Core Features & Modules

### The GPU Frontier (Triton)
Optimized for the NVIDIA H100's Asynchronous Dataflow Engine.
* **Tensor Memory Accelerator (TMA):** Utilizes `tl.make_block_ptr` to issue asynchronous loads from High Bandwidth Memory (HBM) directly to Shared Memory (SRAM).
* **WGMMA Optimization:** Uses 128-thread warpgroups to increase arithmetic intensity and execute dense matrix multiplications.
* **L2 Cache Swizzling:** Custom space-filling logic to prevent partition camping and cache thrashing.

### The TPU Frontier (JAX Pallas)
Optimized for Google's Systolic Dataflow Engine.
* **Explicit DMA Pipelines:** Uses explicit `BlockSpec` definitions and `num_pipeline_stages=2` to hide memory latency.
* **Vector Register Management:** Tuned tile sizes (128x128) to prevent the XLA compiler from spilling Vector Processing Unit (VPU) operations to VMEM.

### Distributed Scaling: Ring Attention
* **Mesh Topology:** Utilizes JAX's `shard_map` (SPMD) to distribute sequence lengths across N devices.
* **Computation-Communication Overlap:** Leverages `lax.ppermute` to rotate Key/Value blocks across the TPU's 3D Torus interconnect while simultaneously computing local attention.

### Precision Engineering: FP8 Quantization
* **Dual Formats:** Supports `E4M3` (forward pass/weights) and `E5M2` (backward pass/gradients) to balance precision and dynamic range.
* **Block-Wise Quantization (Microscaling):** Computes isolated scale factors for every 128x128 block to prevent gradient outlier spikes from collapsing tensor resolution.
* **Stochastic Rounding:** Injects random noise prior to truncation to preserve statistical convergence.

---

## 4. Repository Structure
Based on the strategic blueprint.

```text
omni_attention/
├── layers/
│   ├── attention.py       # High-level JAX/Flax and PyTorch Unified Modules
│   └── ring_dist.py       # shard_map distributed logic
├── kernels/
│   ├── triton/
│   │   ├── flash_attn_v3.py   # Hopper-optimized TMA/WGMMA Kernel
│   │   └── fp8_quant.py       # Block-wise FP8 quantization
│   └── pallas/
│       ├── tpu_flash.py       # v5e Kernel with explicit BlockSpecs
│       └── tpu_layout.py      # Layout transformation logic
├── profiling/
│   ├── benchmark_mfu.py   # MFU calculations (targeting >60%)
│   └── roofline_plot.py   # Roofline analysis scripts
├── serving/               # C++ Infrastructure Layer
│   ├── include/
│   │   └── omni_serve.h       # C++ Runtime Definitions
│   └── src/
│       ├── pjrt_client.cc     # XLA compilation and device management
│       ├── scheduler.cc       # Continuous Batching & Paged Attention
│       └── custom_ops.cc      # XLA Custom Call implementations (Top-P)
└── tests/
    ├── test_equivalence.py    # GPU vs TPU mathematical parity
    └── test_numerics.py       # FP8 vs FP32 validation (RMSE)
```

---

## 5. Installation & Setup

**Prerequisites:**
* NVIDIA GPU (Hopper Architecture recommended for full TMA features) OR Google Cloud TPU (v5e/v4).
* Python 3.10+
* Bazel (for compiling the C++ serving stack)

**Python Environment:**
```bash
python -m venv omni_env
source omni_env/bin/activate

# Install PyTorch with CUDA support (for Triton)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install triton

# Install JAX for TPU (or CPU emulation for testing)
pip install -U "jax[tpu]" -f [https://storage.googleapis.com/jax-releases/libtpu_releases.html](https://storage.googleapis.com/jax-releases/libtpu_releases.html)
pip install flax
```

---

## 6. Usage & Testing

The Python test suite validates the integrity of the mathematical operations across architectures.

```bash
# 1. Verify FP8 Microscaling and Stochastic Rounding
python -m tests.test_numerics

# 2. Verify Cross-Hardware Parity (GPU output == TPU output)
python -m tests.test_equivalence
```

To run the distributed multi-device Ring Attention simulation:
```bash
python -m layers.ring_dist
```

---

## 7. Omni-Serve C++ Infrastructure
While Python is the language of research, C++ is the language of production infrastructure. Omni-Serve is a lightweight C++ inference runtime that bypasses the Python GIL.

**Key Components:**
1.  **PJRT Client (`pjrt_client.cc`):** Controls accelerators directly via the Pre-JIT Runtime API, loading pre-compiled `StableHLO` modules.
2.  **Continuous Batching (`scheduler.cc`):** Manages a custom Block Table mapping logical tokens to physical memory to enable high-throughput asynchronous request handling.
3.  **XLA Custom Calls (`custom_ops.cc`):** Extends the compiler with specialized C++ operations (like Top-P Token Sampling) registered directly into the XLA graph execution flow.

*(Note: Building the C++ stack requires integration with the standard TensorFlow/XLA Bazel toolchain).*

---

## 8. Profiling & Verification
Transforming theoretical logic into a validated engineering artifact requires rigorous profiling.

**Model Flops Utilization (MFU):**
The ultimate measure of success. A standard PyTorch implementation might hit 30-40%; Omni-Attention targets 60-75% on H100 by exploiting TMA/WGMMA overlap.
```bash
python -m profiling.benchmark_mfu
```

**Roofline Analysis:**
Generate operational intensity (FLOPs/Byte) metrics to determine if the kernel is Compute-Bound or Memory-Bound.
```bash
python -m profiling.roofline_plot
```

**Hardware Profilers:**
* **GPU:** Use `ncu` (Nsight Compute) to inspect Shared Memory Bank Conflicts and TMA pipeline efficiency.
* **TPU:** Use JAX Profiler / Trace Viewer to visualize Computation-Communication overlap and ensure HBM bandwidth is saturated.

---

## 9. References
The theoretical frameworks and hardware optimizations in this repository are deeply informed by the Omni-Attention blueprint and the surrounding ecosystem of foundational research.

* **Google TPU & XLA:** Building production AI on Cloud TPUs with JAX, Using JAX to accelerate our research.
* **NVIDIA H100 & Triton:** FlashAttention-3: Fast and Accurate Attention with Asynchrony, Triton Kernel Optimization.
* **Distributed Training:** SPMD Parallelism with `jax.shard_map`, Ring Attention with Blockwise Transformers.
* **Quantization:** FP8 Basics, Block Scaled Matrix Multiplication.