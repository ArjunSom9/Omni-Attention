import numpy as np
import matplotlib.pyplot as plt
import math

# ==============================================================================
# OMNI-ATTENTION: ROOFLINE MODEL ANALYSIS
# Project Phase 3: Profiling & Verification
# Architecture Targets: NVIDIA H100 (SXM) & Google TPU v5e
# Metric: Operational Intensity (FLOPs/Byte) vs. Performance (TFLOPS)
# ==============================================================================

class HardwareAccelerator:
    def __init__(self, name: str, peak_tflops: float, mem_bandwidth_tb_s: float):
        self.name = name
        # Flat part of the roofline (Compute Bound)
        self.peak_tflops = peak_tflops 
        # Slanted part of the roofline (Memory Bound)
        self.mem_bandwidth_tb_s = mem_bandwidth_tb_s 
        
        # The "Ridge Point": The exact Operational Intensity where the bottleneck
        # shifts from Memory Bandwidth to Math/Compute throughput.
        self.ridge_point = peak_tflops / mem_bandwidth_tb_s

# Hardware Specifications (Approximate FP16/BF16 metrics)
H100_SXM = HardwareAccelerator(name="NVIDIA H100 (Hopper)", peak_tflops=989.0, mem_bandwidth_tb_s=3.35)
TPU_V5E = HardwareAccelerator(name="Google TPU v5e", peak_tflops=197.0, mem_bandwidth_tb_s=0.82)


def calculate_flash_attention_metrics(seq_len: int, head_dim: int, causal: bool = False):
    """
    Calculates the theoretical Operational Intensity of FlashAttention.
    
    Standard Attention:
    1. S = Q * K^T  (Batch, Heads, Seq, Seq)
    2. P = Softmax(S)
    3. O = P * V    (Batch, Heads, Seq, HeadDim)
    
    FLOPs: 
    - QK^T: 2 * seq_len * seq_len * head_dim
    - PV:   2 * seq_len * seq_len * head_dim
    Total = 4 * seq_len^2 * head_dim (Halved if causal masking applies)
    
    Bytes Accessed (HBM):
    - Read Q, K, V: 3 * seq_len * head_dim * 2 bytes (FP16)
    - Write O: 1 * seq_len * head_dim * 2 bytes (FP16)
    Total = 8 * seq_len * head_dim
    """
    # Total FLOPs per head
    flops = 4 * (seq_len ** 2) * head_dim
    if causal:
        flops /= 2.0
        
    # Total HBM Bytes per head (Assuming perfect FlashAttention TMA block tiling)
    bytes_accessed = 8 * seq_len * head_dim
    
    operational_intensity = flops / bytes_accessed
    return operational_intensity, flops, bytes_accessed


def plot_roofline(hardware: HardwareAccelerator, observed_tflops: float, op_intensity: float, save_path: str = None):
    """
    Generates a matplotlib Roofline plot and diagnostics.
    """
    print(f"\n--- Roofline Analysis: {hardware.name} ---")
    print(f"Ridge Point: {hardware.ridge_point:.2f} FLOPs/Byte")
    print(f"Kernel Operational Intensity: {op_intensity:.2f} FLOPs/Byte")
    
    # 1. Diagnostics (Section 7.1)
    is_memory_bound = op_intensity < hardware.ridge_point
    
    if is_memory_bound:
        print("\nSTATUS: MEMORY BOUND (Under the slanted roof)")
        print("Diagnosis: The kernel is starving for data from HBM.")
        print("Fix 1: Improve TMA pipelining (increase `num_stages` in Triton).")
        print("Fix 2: Increase BLOCK_M and BLOCK_N tile sizes to increase data reuse in SRAM.")
    else:
        print("\nSTATUS: COMPUTE BOUND (Under the flat roof)")
        print("Diagnosis: The kernel is limited by Tensor Core / MXU throughput.")
        print("Fix 1: Ensure WGMMA instructions are fully unrolled without Math Pipe Throttles.")
        print("Fix 2: Verify that ALU ops (Softmax) are properly interleaving with Tensor Cores.")

    # 2. Plotting
    x = np.logspace(0, 4, 100)
    
    # Calculate the Roofline boundaries
    y_memory_bound = x * hardware.mem_bandwidth_tb_s
    y_compute_bound = np.full_like(x, hardware.peak_tflops)
    y_roof = np.minimum(y_memory_bound, y_compute_bound)
    
    plt.figure(figsize=(10, 6))
    
    # Plot Theoretical Roofline
    plt.loglog(x, y_roof, label=f'{hardware.name} Theoretical Limit', color='black', linewidth=2)
    
    # Plot Observed Performance
    plt.scatter([op_intensity], [observed_tflops], color='red', s=100, zorder=5, label='Omni-Attention Kernel')
    
    # Annotations & Formatting
    plt.axvline(x=hardware.ridge_point, color='gray', linestyle='--', label=f'Ridge Point ({hardware.ridge_point:.1f})')
    
    plt.title(f'Roofline Model: Omni-Attention on {hardware.name}')
    plt.xlabel('Operational Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOPS)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        # plt.show() bypassed for headless execution, 
        # but the diagnostic logic prints the critical conclusions.
        pass


# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # Test Parameters
    SEQ_LEN = 2048
    HEAD_DIM = 128
    
    # Calculate physics based on the math of the algorithm
    intensity, _, _ = calculate_flash_attention_metrics(seq_len=SEQ_LEN, head_dim=HEAD_DIM, causal=False)
    
    print("Omni-Attention Roofline Profiler")
    print("="*40)
    print(f"Sequence Length: {SEQ_LEN}, Head Dimension: {HEAD_DIM}")
    
    # -------------------------------------------------------------------------
    # Scenario A: Evaluating the H100 Kernel
    # -------------------------------------------------------------------------
    # Let's assume Nsight Compute (ncu) reported our Triton kernel hitting 550 TFLOPS
    OBSERVED_H100_TFLOPS = 550.0 
    plot_roofline(H100_SXM, OBSERVED_H100_TFLOPS, intensity, save_path="h100_roofline.png")
    
    # -------------------------------------------------------------------------
    # Scenario B: Evaluating the TPU v5e Kernel
    # -------------------------------------------------------------------------
    # Let's assume JAX Profiler reported our Pallas kernel hitting 145 TFLOPS
    OBSERVED_TPU_TFLOPS = 145.0
    plot_roofline(TPU_V5E, OBSERVED_TPU_TFLOPS, intensity, save_path="tpu_v5e_roofline.png")