import math
import jax
import jax.numpy as jnp
import flax.linen as nn
import torch
import torch.nn as py_nn

# ==============================================================================
# OMNI-ATTENTION: HIGH-LEVEL COMPOSITION LAYER
# Project Phase 2: Unifying Hardware Kernels into Standard Neural Modules
# ==============================================================================

# Attempt to import our custom hardware kernels. 
# In a real environment, these are in the `kernels/` directory.
try:
    from kernels.pallas.tpu_flash import call_pallas_flash
except ImportError:
    call_pallas_flash = None
    print("Warning: TPU Pallas kernel not found. Flax module will use fallback.")

try:
    from kernels.triton.flash_attn_v3 import flash_attn_v3_forward
    from kernels.triton.fp8_quant import block_quantize_fp8
except ImportError:
    flash_attn_v3_forward = None
    block_quantize_fp8 = None
    print("Warning: Triton GPU kernels not found. PyTorch module will use fallback.")


# ------------------------------------------------------------------------------
# 1. THE TPU STACK (JAX / FLAX)
# ------------------------------------------------------------------------------
class OmniAttentionFlax(nn.Module):
    """
    High-level Flax module targeting Google Cloud TPUs.
    Wraps the Pallas explicitly-pipelined systolic array kernel.
    """
    num_heads: int
    head_dim: int
    
    @nn.compact
    def __call__(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
            
        batch, seq_len, _ = x_q.shape
        
        # Linear projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='q_proj')(x_q)
        k = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='k_proj')(x_kv)
        v = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='v_proj')(x_kv)
        
        # Reshape to (Batch, Heads, SeqLen, HeadDim) for the Pallas Kernel
        # JAX makes it easy to transpose without physical memory copies prior to the kernel
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        k = k.reshape(batch, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose((0, 2, 1, 3))
        
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        
        # Dispatch to our custom Pallas Kernel
        if call_pallas_flash is not None:
            attn_out = call_pallas_flash(q, k, v, sm_scale=sm_scale)
        else:
            # Standard JAX fallback for testing without TPU hardware
            scores = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * sm_scale
            weights = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.matmul(weights, v)
            
        # Reshape back to (Batch, SeqLen, ModelDim)
        attn_out = attn_out.transpose((0, 2, 1, 3)).reshape(batch, seq_len, -1)
        
        # Output projection
        out = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='out_proj')(attn_out)
        return out


# ------------------------------------------------------------------------------
# 2. THE GPU STACK (PYTORCH)
# ------------------------------------------------------------------------------
class OmniAttentionTorch(py_nn.Module):
    """
    High-level PyTorch module targeting NVIDIA Hopper.
    Wraps the Triton TMA/WGMMA kernel and conditionally applies FP8 quantization.
    """
    def __init__(self, d_model: int, num_heads: int, use_fp8: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_fp8 = use_fp8
        
        self.q_proj = py_nn.Linear(d_model, d_model, bias=False)
        self.k_proj = py_nn.Linear(d_model, d_model, bias=False)
        self.v_proj = py_nn.Linear(d_model, d_model, bias=False)
        self.out_proj = py_nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x_q, x_kv=None):
        if x_kv is None:
            x_kv = x_q
            
        batch, seq_len, _ = x_q.shape
        
        # 1. Optional FP8 Quantization (Microscaling) for memory bandwidth savings
        if self.use_fp8 and block_quantize_fp8 is not None:
            # We quantize the activations to E4M3 before projection
            x_q_fp8, _ = block_quantize_fp8(x_q, fp8_dtype=torch.float8_e4m3fn)
            x_kv_fp8, _ = block_quantize_fp8(x_kv, fp8_dtype=torch.float8_e4m3fn)
            # In a full implementation, the Linear layers would be custom FP8 GEMMs
            q = self.q_proj(x_q_fp8.to(x_q.dtype)) 
            k = self.k_proj(x_kv_fp8.to(x_kv.dtype))
            v = self.v_proj(x_kv_fp8.to(x_kv.dtype))
        else:
            q = self.q_proj(x_q)
            k = self.k_proj(x_kv)
            v = self.v_proj(x_kv)

        # Reshape to (Batch, Heads, SeqLen, HeadDim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        
        # 2. Dispatch to custom Triton H100 Kernel
        if flash_attn_v3_forward is not None:
            attn_out = flash_attn_v3_forward(q, k, v, sm_scale)
        else:
            # Standard PyTorch fallback (SDPA)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=sm_scale
            )
            
        # Reshape back and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.out_proj(attn_out)
        
        return out


# ==============================================================================
# USAGE / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    BATCH = 2
    SEQ_LEN = 128
    D_MODEL = 512
    NUM_HEADS = 8
    
    print("Omni-Attention Layer Unified API Test\n" + "="*40)
    
    # --- Test PyTorch (GPU) Module ---
    print("1. Initializing PyTorch (GPU) Instance...")
    pt_model = OmniAttentionTorch(d_model=D_MODEL, num_heads=NUM_HEADS, use_fp8=False)
    
    x_pt = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out_pt = pt_model(x_pt)
    print(f"   [PyTorch] Output Shape: {out_pt.shape}")
    
    # --- Test Flax (TPU) Module ---
    print("\n2. Initializing Flax (TPU) Instance...")
    rng = jax.random.PRNGKey(0)
    flax_model = OmniAttentionFlax(num_heads=NUM_HEADS, head_dim=D_MODEL // NUM_HEADS)
    
    x_jax = jnp.ones((BATCH, SEQ_LEN, D_MODEL))
    variables = flax_model.init(rng, x_jax)
    out_jax = flax_model.apply(variables, x_jax)
    print(f"   [Flax]    Output Shape: {out_jax.shape}")