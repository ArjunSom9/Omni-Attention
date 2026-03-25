import torch
import triton
import triton.language as tl

# ==============================================================================
# OMNI-ATTENTION: FP8 PRECISION ENGINEERING (MICROSCALING)
# Architecture Target: NVIDIA H100 (Hopper)
# Core Mechanics: E4M3/E5M2 dual formats, Block-wise Scaling, Stochastic Rounding
# ==============================================================================

@triton.jit
def _fp8_block_quantize_kernel(
    # Data Pointers
    x_ptr,          # Input FP32/BF16 tensor
    out_ptr,        # Output FP8 tensor
    scale_ptr,      # Output FP32 scale factors (one per block)
    # Strides
    stride_xm, stride_xn,
    stride_om, stride_on,
    stride_sm, stride_sn,
    # Dimensions
    M, N,
    # Constants
    FP8_MAX: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    """
    Triton Kernel for Block-Wise FP8 Quantization.
    Each Thread Block (CTA) processes a (BLOCK_M, BLOCK_N) tile of the matrix,
    finds the local maximum, calculates a localized scale factor, applies it,
    optionally performs stochastic rounding, and casts to FP8.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 1. 2D Pointer Arithmetic for the Block
    # We use standard offsets here to easily coordinate with the PRNG 
    # required for stochastic rounding.
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    
    # Mask to handle matrices that aren't perfect multiples of BLOCK_M/N
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # 2. Load the tile
    x = tl.load(x_ptrs, mask=mask, other=0.0)

    # 3. Compute Block-Local Scale Factor
    # Scale = FP8_MAX / Max(Abs(Tensor_block))
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x)  # Triton reduces this 2D block to a single scalar
    
    # Avoid division by zero for empty/zero blocks
    safe_max = tl.maximum(max_val, 1e-12)
    
    # We store the *divisor* as the scale (Standard practice in microscaling)
    # scale_factor = max_val / FP8_MAX
    scale_factor = safe_max / FP8_MAX
    inv_scale = 1.0 / scale_factor

    # Apply scaling
    x_scaled = x * inv_scale

    # 4. Stochastic Rounding (Section 6.3 Numerics and Stability)
    # Truncation introduces directional bias. By adding uniform noise [-0.5, 0.5]
    # prior to the hardware cast, the *expected value* of the FP8 tensor remains
    # exactly equal to the higher precision tensor.
    if STOCHASTIC:
        # Seed PRNG uniquely per element based on matrix coordinates
        seed = 1337
        linear_offs = offs_m[:, None] * N + offs_n[None, :]
        noise = tl.rand(seed, linear_offs) - 0.5
        x_scaled = x_scaled + noise

    # 5. Store FP8 Data and FP32 Scale Factor
    # The cast to float8e4nv or float8e5 is handled implicitly by tl.store
    # based on the underlying dtype of out_ptr, but we can explicitly cast it.
    # We just store; the hardware truncates/rounds to nearest even natively.
    tl.store(out_ptrs, x_scaled, mask=mask)

    # Calculate pointer for the 2D scale grid and store the single scale_factor
    s_ptr = scale_ptr + (pid_m * stride_sm + pid_n * stride_sn)
    
    # Only thread 0 in the block needs to write the scale, but for a scalar, 
    # doing a single store across the CTA is safe in Triton.
    tl.store(s_ptr, scale_factor)


# ==============================================================================
# PYTHON WRAPPER (Public API)
# ==============================================================================
def block_quantize_fp8(x: torch.Tensor, fp8_dtype=torch.float8_e4m3fn, stochastic=False):
    """
    Applies block-wise FP8 quantization to a 2D/3D tensor.
    
    Args:
        x (torch.Tensor): The input tensor (FP32, BF16, or FP16).
        fp8_dtype (torch.dtype): torch.float8_e4m3fn (forward) or torch.float8_e5m2 (backward).
        stochastic (bool): Whether to apply stochastic rounding noise.
        
    Returns:
        tuple: (FP8 Tensor, Scale Factors Tensor)
    """
    assert x.is_cuda, "Tensor must be on GPU for Triton kernels"
    assert x.dim() >= 2, "Tensor must be at least 2D"

    # Flatten batch dimensions into M, keeping the last dimension as N
    original_shape = x.shape
    x_2d = x.view(-1, original_shape[-1])
    M, N = x_2d.shape

    # Tuning: 128x128 blocks match the WGMMA instruction layout size on Hopper
    BLOCK_M = 128
    BLOCK_N = 128
    
    grid_m = triton.cdiv(M, BLOCK_M)
    grid_n = triton.cdiv(N, BLOCK_N)

    # Allocate outputs
    out = torch.empty_like(x_2d, dtype=fp8_dtype)
    scales = torch.empty((grid_m, grid_n), device=x.device, dtype=torch.float32)

    # Determine maximum representable value based on FP8 flavor
    if fp8_dtype == torch.float8_e4m3fn:
        # E4M3: 4 bits exponent, 3 bits mantissa (Max ~448.0)
        # Use case: Forward pass weights and activations
        fp8_max = 448.0
    elif fp8_dtype == torch.float8_e5m2:
        # E5M2: 5 bits exponent, 2 bits mantissa (Max ~57344.0)
        # Use case: Backward pass gradients (handles spikes)
        fp8_max = 57344.0
    else:
        raise ValueError(f"Unsupported dtype: {fp8_dtype}. Use float8_e4m3fn or float8_e5m2.")

    grid = (grid_m, grid_n)

    # Launch Kernel
    _fp8_block_quantize_kernel[grid](
        x_2d, out, scales,
        x_2d.stride(0), x_2d.stride(1),
        out.stride(0), out.stride(1),
        scales.stride(0), scales.stride(1),
        M, N,
        FP8_MAX=fp8_max,
        STOCHASTIC=stochastic,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4
    )

    # Reshape the output back to original (e.g., Batch, Seq, Dim)
    return out.view(original_shape), scales


# ==============================================================================
# USAGE / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    # Test Settings
    SEQ_LEN = 4096
    DIM = 4096
    
    print(f"Initializing FP32 Weight Tensor ({SEQ_LEN}x{DIM})...")
    # Simulate a weight matrix with a normal distribution
    weights_fp32 = torch.randn((SEQ_LEN, DIM), device='cuda', dtype=torch.float32)
    
    # Simulate an outlier spike (gradient spike)
    weights_fp32[0, 0] = 1000.0 
    
    try:
        # 1. Forward Pass Strategy: E4M3 (High Precision, Low Range)
        print("\n[Phase 1] Quantizing Weights (E4M3 - Forward Pass)...")
        w_fp8_e4m3, w_scales = block_quantize_fp8(weights_fp32, torch.float8_e4m3fn)
        
        print(f"FP8 Shape: {w_fp8_e4m3.shape}, dtype: {w_fp8_e4m3.dtype}")
        print(f"Scales Grid Shape: {w_scales.shape}")
        
        # 2. Backward Pass Strategy: E5M2 (High Range, Low Precision) + Stochastic Rounding
        print("\n[Phase 2] Quantizing Gradients (E5M2 - Backward Pass) with Stochastic Rounding...")
        g_fp8_e5m2, g_scales = block_quantize_fp8(weights_fp32, torch.float8_e5m2, stochastic=True)
        
        print(f"FP8 Shape: {g_fp8_e5m2.shape}, dtype: {g_fp8_e5m2.dtype}")
        print(f"Scales Grid Shape: {g_scales.shape}")
        
    except Exception as e:
        print(f"Kernel execution failed (Ensure you are on a GPU that supports PyTorch fp8 dtypes): {e}")