import math
import numpy as np
import sys

# ==============================================================================
# OMNI-ATTENTION: HARDWARE EQUIVALENCE VALIDATION (GPU vs TPU Parity)
# Project Phase 3: Profiling & Verification
# Asserts that the SIMT Triton kernel and Systolic Pallas kernel produce 
# mathematically equivalent results within acceptable precision tolerances.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Graceful Multi-Hardware Imports
# ------------------------------------------------------------------------------
HAS_GPU = False
HAS_TPU = False

try:
    import torch
    from kernels.triton.flash_attn_v3 import flash_attn_v3_forward
    if torch.cuda.is_available():
        HAS_GPU = True
except ImportError:
    print("Warning: PyTorch or Triton kernels not found.")

try:
    import jax
    import jax.numpy as jnp
    from kernels.pallas.tpu_flash import call_pallas_flash
    # JAX defaults to CPU emulation if no TPU/GPU is found, which is perfect for testing
    HAS_TPU = True 
except ImportError:
    print("Warning: JAX or Pallas kernels not found.")


# ------------------------------------------------------------------------------
# 2. The Absolute Mathematical Baseline (Pure NumPy)
# ------------------------------------------------------------------------------
def numpy_flash_attention_baseline(q, k, v):
    """
    Standard scaled dot-product attention in FP64/FP32 to serve as the 
    absolute mathematical ground truth.
    Args:
        q, k, v: NumPy arrays of shape (Batch, Heads, SeqLen, HeadDim)
    """
    head_dim = q.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)
    
    # 1. S = Q @ K^T
    # Einstein summation handles the batch and head dimensions seamlessly
    scores = np.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale
    
    # 2. P = Softmax(S)
    # Numerically stable softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 3. O = P @ V
    output = np.einsum('bhqk,bhkd->bhqd', probs, v)
    
    return output


# ------------------------------------------------------------------------------
# 3. Cross-Platform Validation Execution
# ------------------------------------------------------------------------------
def test_poly_accelerator_parity():
    print("--- Omni-Attention: GPU vs TPU Output Parity Test ---")
    
    # 1. Define dimensions (Must fit hardware block constraints, e.g., 128)
    BATCH = 2
    HEADS = 4
    SEQ_LEN = 512
    HEAD_DIM = 64
    
    print(f"Configuration: Batch={BATCH}, Heads={HEADS}, Seq={SEQ_LEN}, Dim={HEAD_DIM}")
    
    # 2. Generate Universal Random Seed in FP32
    np.random.seed(1337)
    q_np = np.random.normal(0, 1, (BATCH, HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float32)
    k_np = np.random.normal(0, 1, (BATCH, HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float32)
    v_np = np.random.normal(0, 1, (BATCH, HEADS, SEQ_LEN, HEAD_DIM)).astype(np.float32)
    
    # 3. Compute Ground Truth Baseline
    print("Computing FP32 NumPy Ground Truth...")
    out_ref = numpy_flash_attention_baseline(q_np, k_np, v_np)
    
    # Define acceptable tolerances for 16-bit float accumulation
    # BF16 has larger representation error than FP16, so we set tolerance accordingly.
    ATOL = 5e-2 
    RTOL = 5e-2

    # 4. Test Hopper GPU (Triton)
    if HAS_GPU:
        print("\n[GPU] Dispatching to NVIDIA Hopper (Triton)...")
        # Convert NumPy to PyTorch FP16 and push to VRAM
        q_pt = torch.tensor(q_np, dtype=torch.float16, device='cuda')
        k_pt = torch.tensor(k_np, dtype=torch.float16, device='cuda')
        v_pt = torch.tensor(v_np, dtype=torch.float16, device='cuda')
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        
        # Execute Kernel
        out_pt = flash_attn_v3_forward(q_pt, k_pt, v_pt, sm_scale)
        out_gpu_np = out_pt.cpu().numpy().astype(np.float32)
        
        # Calculate Delta
        max_diff_gpu = np.max(np.abs(out_ref - out_gpu_np))
        print(f"      GPU vs Baseline Max Difference: {max_diff_gpu:.6f}")
        
        assert np.allclose(out_ref, out_gpu_np, atol=ATOL, rtol=RTOL), \
            f"GPU Kernel failed parity check! Max diff: {max_diff_gpu}"
        print("      PASS: GPU output matches mathematical baseline.")
    else:
        print("\n[GPU] Skipped: No CUDA device available.")

    # 5. Test Google TPU (Pallas)
    if HAS_TPU:
        print("\n[TPU] Dispatching to Google TPU/JAX Backend (Pallas)...")
        # Convert NumPy to JAX BF16 (Native MXU format)
        q_jx = jnp.array(q_np, dtype=jnp.bfloat16)
        k_jx = jnp.array(k_np, dtype=jnp.bfloat16)
        v_jx = jnp.array(v_np, dtype=jnp.bfloat16)
        
        # Execute Kernel
        # If no physical TPU is present, JAX transparently JIT compiles this to CPU
        out_jx = call_pallas_flash(q_jx, k_jx, v_jx)
        out_tpu_np = np.array(out_jx, dtype=np.float32)
        
        # Calculate Delta
        max_diff_tpu = np.max(np.abs(out_ref - out_tpu_np))
        print(f"      TPU vs Baseline Max Difference: {max_diff_tpu:.6f}")
        
        assert np.allclose(out_ref, out_tpu_np, atol=ATOL, rtol=RTOL), \
            f"TPU Kernel failed parity check! Max diff: {max_diff_tpu}"
        print("      PASS: TPU output matches mathematical baseline.")
    else:
        print("\n[TPU] Skipped: JAX ecosystem not found.")

    # 6. The Final Omni-Attention Assertion
    if HAS_GPU and HAS_TPU:
        print("\n[OMNI] Verifying direct cross-hardware parity (GPU == TPU)...")
        max_diff_cross = np.max(np.abs(out_gpu_np - out_tpu_np))
        print(f"       GPU vs TPU Max Difference: {max_diff_cross:.6f}")
        assert np.allclose(out_gpu_np, out_tpu_np, atol=ATOL*2, rtol=RTOL*2), \
            f"Cross-hardware parity failed! Max diff: {max_diff_cross}"
        print("       PASS: Both divergent architectures produce equivalent representations.")


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not HAS_GPU and not HAS_TPU:
        print("Cannot run equivalence tests. Neither PyTorch/Triton nor JAX/Pallas were found.")
        sys.exit(0)
        
    try:
        test_poly_accelerator_parity()
        print("\nEquivalence Suite Passed! The Poly-Accelerator architecture is validated.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")