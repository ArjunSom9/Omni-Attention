import math
import torch
import sys

# ==============================================================================
# OMNI-ATTENTION: NUMERICS AND STABILITY VALIDATION
# Project Phase 3: Profiling & Verification
# Validating FP8 Microscaling (Block-wise Quantization) against FP32 Baselines
# ==============================================================================

# Attempt to import our custom hardware kernels
try:
    from kernels.triton.fp8_quant import block_quantize_fp8
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("Warning: Triton GPU kernels not found. Tests will be skipped if not on GPU.")

# Hardware specific block dimensions defined in our kernel
BLOCK_M = 128
BLOCK_N = 128


def calculate_rmse(baseline: torch.Tensor, target: torch.Tensor) -> float:
    """Calculates the Root Mean Square Error between two tensors."""
    mse = torch.nn.functional.mse_loss(baseline.float(), target.float())
    return math.sqrt(mse.item())


def dequantize_block_fp8(fp8_tensor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantizes the FP8 tensor back to FP32 using the block-wise scale grid.
    This simulates what the Tensor Cores do implicitly during the WGMMA dot product.
    """
    original_shape = fp8_tensor.shape
    x_2d = fp8_tensor.view(-1, original_shape[-1])
    
    # Expand the (Grid_M, Grid_N) scale matrix back to the (M, N) tensor shape
    # by repeating the scale factors across the 128x128 blocks.
    expanded_scales = scales.repeat_interleave(BLOCK_M, dim=0).repeat_interleave(BLOCK_N, dim=1)
    
    # Crop to exact shape in case of padding
    expanded_scales = expanded_scales[:x_2d.shape[0], :x_2d.shape[1]]
    
    # Dequantize: X_fp32 = X_fp8 * Scale
    dequantized_2d = x_2d.float() * expanded_scales
    
    return dequantized_2d.view(original_shape)


# ------------------------------------------------------------------------------
# TEST 1: E4M3 Forward Pass (High Precision, Low Range)
# ------------------------------------------------------------------------------
def test_e4m3_precision():
    """Validates that standard normally-distributed weights map cleanly to E4M3."""
    print("--- Test 1: E4M3 Precision (Forward Pass) ---")
    torch.manual_seed(42)
    
    # Simulate a weight matrix
    x_fp32 = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
    
    # Quantize
    x_fp8, scales = block_quantize_fp8(x_fp32, fp8_dtype=torch.float8_e4m3fn, stochastic=False)
    
    # Dequantize and Measure
    x_dequant = dequantize_block_fp8(x_fp8, scales)
    rmse = calculate_rmse(x_fp32, x_dequant)
    
    print(f"E4M3 RMSE: {rmse:.5f}")
    assert rmse < 0.05, f"E4M3 quantization error too high! RMSE: {rmse}"
    print("PASS: E4M3 error is within the acceptable model noise floor.\n")


# ------------------------------------------------------------------------------
# TEST 2: E5M2 Backward Pass (High Range, Outlier Resistance)
# ------------------------------------------------------------------------------
def test_e5m2_outlier_handling():
    """Validates that E5M2 can handle massive gradient spikes without overflowing."""
    print("--- Test 2: E5M2 Outlier Handling (Backward Pass) ---")
    torch.manual_seed(42)
    
    # Simulate gradients
    g_fp32 = torch.randn((1024, 1024), device='cuda', dtype=torch.float32)
    
    # Inject massive outliers that would overflow E4M3 (Max ~448.0)
    g_fp32[0, 0] = 5000.0
    g_fp32[128, 128] = -4500.0
    
    # Quantize
    g_fp8, scales = block_quantize_fp8(g_fp32, fp8_dtype=torch.float8_e5m2, stochastic=False)
    
    # Verify no NaNs or Infs
    g_fp32_recovered = dequantize_block_fp8(g_fp8, scales)
    max_recovered = g_fp32_recovered.abs().max().item()
    
    print(f"Max original value: {g_fp32.abs().max().item():.1f}")
    print(f"Max recovered value: {max_recovered:.1f}")
    
    assert not torch.isnan(g_fp32_recovered).any(), "E5M2 quantization resulted in NaNs!"
    assert not torch.isinf(g_fp32_recovered).any(), "E5M2 quantization resulted in Infs!"
    assert max_recovered > 4000.0, "E5M2 failed to preserve the outlier spike magnitude."
    print("PASS: E5M2 successfully preserved high-dynamic-range outliers without divergence.\n")


# ------------------------------------------------------------------------------
# TEST 3: Stochastic Rounding Bias Analysis
# ------------------------------------------------------------------------------
def test_stochastic_rounding_bias():
    """Validates that stochastic rounding centers the error mean at 0."""
    print("--- Test 3: Stochastic Rounding Bias Analysis ---")
    torch.manual_seed(42)
    
    # Create a uniform tensor where standard truncation causes a distinct directional bias
    x_fp32 = torch.rand((2048, 2048), device='cuda', dtype=torch.float32) * 2.0 - 1.0
    
    # 1. Standard Quantization (Truncation/Nearest)
    x_standard, scales_std = block_quantize_fp8(x_fp32, fp8_dtype=torch.float8_e4m3fn, stochastic=False)
    x_dequant_std = dequantize_block_fp8(x_standard, scales_std)
    error_std = (x_fp32 - x_dequant_std)
    mean_error_std = error_std.mean().item()
    
    # 2. Stochastic Quantization
    x_stochastic, scales_stoch = block_quantize_fp8(x_fp32, fp8_dtype=torch.float8_e4m3fn, stochastic=True)
    x_dequant_stoch = dequantize_block_fp8(x_stochastic, scales_stoch)
    error_stoch = (x_fp32 - x_dequant_stoch)
    mean_error_stoch = error_stoch.mean().item()
    
    print(f"Mean Error (Standard Truncation): {mean_error_std:.7f}")
    print(f"Mean Error (Stochastic Rounding): {mean_error_stoch:.7f}")
    
    # The stochastic mean error should be strictly closer to 0 than the standard error
    assert abs(mean_error_stoch) < abs(mean_error_std), "Stochastic rounding failed to reduce bias!"
    print("PASS: Stochastic rounding successfully mitigated truncation bias.\n")


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    if not torch.cuda.is_available() or not HAS_TRITON:
        print("Skipping tests: Requires CUDA GPU and Triton installed.")
        sys.exit(0)
        
    print("Starting FP8 Numerics Verification Suite...\n")
    try:
        test_e4m3_precision()
        test_e5m2_outlier_handling()
        test_stochastic_rounding_bias()
        print("All Numerics Tests Passed Successfully! The kernel is stable for training.")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")