import math
import time
import sys

# ==============================================================================
# OMNI-ATTENTION: MODEL FLOPS UTILIZATION (MFU) BENCHMARK
# Project Phase 3: Profiling & Verification
# Measures the raw TFLOPS of our kernels and compares them to hardware limits.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Hardware Theoretical Limits (BF16 / FP16 Dense Math)
# ------------------------------------------------------------------------------
# NVIDIA H100 SXM5: ~989 TFLOPS (FP16/BF16 without structural sparsity)
H100_PEAK_TFLOPS = 989.0 

# Google TPU v5e: ~197 TFLOPS (BF16)
TPU_V5E_PEAK_TFLOPS = 197.0 

# ------------------------------------------------------------------------------
# 2. FLOPs Math for Attention
# ------------------------------------------------------------------------------
def calculate_attention_flops(batch: int, heads: int, seq_len: int, head_dim: int, causal: bool = False):
    """
    Calculates the total mathematical operations for standard Attention.
    Q * K^T = 2 * B * H * S^2 * D
    P * V   = 2 * B * H * S^2 * D
    Total   = 4 * B * H * S^2 * D
    (Divided by 2 if using causal masking)
    """
    flops = 4 * batch * heads * (seq_len ** 2) * head_dim
    if causal:
        flops /= 2.0
    return flops


# ------------------------------------------------------------------------------
# 3. NVIDIA H100 (Triton) Benchmark
# ------------------------------------------------------------------------------
def benchmark_gpu_triton(batch, heads, seq_len, head_dim, warmup=10, iters=100):
    try:
        import torch
        from kernels.triton.flash_attn_v3 import flash_attn_v3_forward
    except ImportError:
        print("[GPU] Triton kernel or PyTorch not found. Skipping.")
        return

    if not torch.cuda.is_available():
        print("[GPU] No CUDA device available. Skipping.")
        return
        
    print(f"\n--- Benchmarking NVIDIA H100 (Triton) ---")
    
    # Allocate Tensors
    q = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device='cuda', dtype=torch.float16)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Warmup (Compiles Triton PTX and ramps up GPU clocks)
    for _ in range(warmup):
        _ = flash_attn_v3_forward(q, k, v, sm_scale)
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = flash_attn_v3_forward(q, k, v, sm_scale)
    end_event.record()
    torch.cuda.synchronize()

    # Calculate MFU
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_s = (total_time_ms / 1000.0) / iters
    
    flops_per_iter = calculate_attention_flops(batch, heads, seq_len, head_dim)
    tflops_achieved = (flops_per_iter / avg_time_s) / (10**12)
    mfu = (tflops_achieved / H100_PEAK_TFLOPS) * 100

    print(f"Average Latency: {avg_time_s * 1000:.3f} ms")
    print(f"Observed TFLOPS: {tflops_achieved:.1f} TFLOPS")
    print(f"H100 Peak TFLOPS: {H100_PEAK_TFLOPS:.1f} TFLOPS")
    print(f"Model Flops Utilization (MFU): {mfu:.1f}%")
    
    # Target Evaluation (From Section 7.3)
    if mfu >= 60.0:
        print("Verdict: EXCELLENT. Meets highly-optimized FlashAttention-3 targets (>60%).")
    elif mfu >= 30.0:
        print("Verdict: AVERAGE. Matches standard library performance (30-40%). Check pipelining.")
    else:
        print("Verdict: POOR. Kernel is severely bottlenecked. Run `ncu` roofline analysis.")


# ------------------------------------------------------------------------------
# 4. Google TPU v5e (Pallas) Benchmark
# ------------------------------------------------------------------------------
def benchmark_tpu_pallas(batch, heads, seq_len, head_dim, warmup=5, iters=50):
    try:
        import jax
        import jax.numpy as jnp
        from kernels.pallas.tpu_flash import call_pallas_flash
    except ImportError:
        print("[TPU] JAX or Pallas kernel not found. Skipping.")
        return
        
    print(f"\n--- Benchmarking Google TPU v5e (Pallas) ---")
    
    # Note: On a non-TPU machine, this will benchmark the CPU emulator.
    # We proceed anyway to demonstrate the methodology.
    devices = jax.devices()
    device_type = devices[0].device_kind
    print(f"Detected Hardware: {device_type}")
    
    q = jax.random.normal(jax.random.PRNGKey(0), (batch, heads, seq_len, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(1), (batch, heads, seq_len, head_dim), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(2), (batch, heads, seq_len, head_dim), dtype=jnp.bfloat16)

    # Warmup (Triggers XLA compilation)
    # block_until_ready() is critical in JAX to force async execution to finish
    print("Compiling Pallas Kernel via XLA...")
    for _ in range(warmup):
        _ = call_pallas_flash(q, k, v).block_until_ready()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = call_pallas_flash(q, k, v).block_until_ready()
    end_time = time.perf_counter()

    # Calculate MFU
    avg_time_s = (end_time - start_time) / iters
    flops_per_iter = calculate_attention_flops(batch, heads, seq_len, head_dim)
    tflops_achieved = (flops_per_iter / avg_time_s) / (10**12)
    
    # If running on CPU emulation, MFU vs TPU_PEAK is meaningless, but we calculate it anyway
    mfu = (tflops_achieved / TPU_V5E_PEAK_TFLOPS) * 100

    print(f"Average Latency: {avg_time_s * 1000:.3f} ms")
    print(f"Observed TFLOPS: {tflops_achieved:.1f} TFLOPS")
    
    if 'TPU' in device_type.upper():
        print(f"TPU v5e Peak TFLOPS: {TPU_V5E_PEAK_TFLOPS:.1f} TFLOPS")
        print(f"Model Flops Utilization (MFU): {mfu:.1f}%")
    else:
        print(f"(MFU calculation skipped. Emulated on {device_type}, not physical TPU)")


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Workload Shape (Optimized for both H100 warpgroups and TPU MXUs)
    BATCH = 4
    HEADS = 32
    SEQ_LEN = 4096
    HEAD_DIM = 128
    
    print("Omni-Attention: Commencing MFU Benchmarks")
    print("="*50)
    print(f"Workload: Batch={BATCH}, Heads={HEADS}, SeqLen={SEQ_LEN}, HeadDim={HEAD_DIM}")
    flops_req = calculate_attention_flops(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    print(f"Total FLOPs per Forward Pass: {flops_req / (10**9):.2f} GFLOPs")
    
    benchmark_gpu_triton(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    benchmark_tpu_pallas(BATCH, HEADS, SEQ_LEN, HEAD_DIM)
    
    print("\nBenchmarking Complete.")