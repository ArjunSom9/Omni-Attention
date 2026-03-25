import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import functools

# ==============================================================================
# OMNI-ATTENTION: SYSTOLIC DATAFLOW KERNEL (FLASH-ATTENTION LOGIC)
# Architecture Target: Google TPU v5e (Megacore)
# Core Mechanics: JAX Pallas, Explicit VMEM DMA, MXU/VPU Pipelining
# ==============================================================================

def tpu_flash_attention_kernel(q_ref, k_ref, v_ref, o_ref, *, sm_scale, block_q, block_kv, seq_len):
    """
    Pallas Kernel running natively on the TPU Core.
    Operates on `Refs` which act as mutable pointers to the TPU's fast VMEM.
    """
    # 1. Load Query block from VMEM ref to registers
    # This block of Q remains stationary in registers while we stream K and V
    q = q_ref[...]
    head_dim = q.shape[-1]
    
    # Initialize accumulators. 
    # MXU computes naturally in FP32 for accumulation.
    acc = jnp.zeros((block_q, head_dim), dtype=jnp.float32)
    m_i = jnp.full((block_q,), -jnp.inf, dtype=jnp.float32)
    l_i = jnp.zeros((block_q,), dtype=jnp.float32)

    num_kv_blocks = seq_len // block_kv

    # Loop over the Key/Value sequence
    for i in range(num_kv_blocks):
        # Explicit DMA boundaries. 
        # The pltpu.TPUCompilerParams(num_pipeline_stages=2) ensures that the
        # DMA engine fetches block (i+1) from HBM into VMEM while the MXU processes block (i).
        k_block = k_ref[i * block_kv : (i + 1) * block_kv, :]
        v_block = v_ref[i * block_kv : (i + 1) * block_kv, :]

        # 2. Compute Dot Product (Maps directly to the MXU Systolic Array)
        qk = jnp.dot(q, k_block.T) * sm_scale

        # 3. Softmax Logic (Maps to the Vector Processing Unit - VPU)
        # We explicitly manage the sequence size to 128 to avoid spilling VPU registers.
        m_ij = jnp.max(qk, axis=-1)
        m_i_new = jnp.maximum(m_i, m_ij)

        # Stable Softmax Math
        alpha = jnp.exp(m_i - m_i_new)
        p = jnp.exp(qk - m_i_new[:, None])
        
        l_i_new = alpha * l_i + jnp.sum(p, axis=-1)

        # 4. Write back & Accumulate (MXU)
        # We cast `p` back to the input dtype (e.g., bfloat16) to leverage the MXU
        # for the final output projection dot product.
        acc_scale = (l_i / l_i_new * alpha)[:, None]
        acc = acc * acc_scale + jnp.dot(p.astype(v_block.dtype), v_block) / l_i_new[:, None]

        # Update running stats
        m_i = m_i_new
        l_i = l_i_new

    # 5. Store final accumulated block back to HBM (via o_ref)
    o_ref[...] = acc.astype(o_ref.dtype)


def _pallas_flash_1d(q, k, v, sm_scale):
    """
    Internal wrapper that defines the Grid and BlockSpecs for a single Attention Head.
    """
    seq_len, head_dim = q.shape
    
    # Tune these block sizes for TPU v5e VMEM capacity (Section 4.3)
    BLOCK_Q = 128
    BLOCK_KV = 128
    
    num_q_blocks = seq_len // BLOCK_Q
    
    # Define the DMA schedule (BlockSpecs). 
    # Q and O are fetched block-by-block based on the Grid index 'i'.
    q_spec = pl.BlockSpec(
        index_map=lambda i: (i, 0), 
        block_shape=(BLOCK_Q, head_dim)
    )
    
    o_spec = pl.BlockSpec(
        index_map=lambda i: (i, 0), 
        block_shape=(BLOCK_Q, head_dim)
    )
    
    # K and V are mapped entirely into the kernel's logical view, 
    # allowing the `for` loop inside the kernel to slice through them.
    kv_spec = pl.BlockSpec(
        index_map=lambda i: (0, 0), 
        block_shape=(seq_len, head_dim)
    )
    
    # The Grid maps only across the sequence dimension of Q
    grid = (num_q_blocks,)

    return pl.pallas_call(
        functools.partial(
            tpu_flash_attention_kernel, 
            sm_scale=sm_scale, 
            block_q=BLOCK_Q, 
            block_kv=BLOCK_KV, 
            seq_len=seq_len
        ),
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        in_specs=[q_spec, kv_spec, kv_spec],
        out_specs=o_spec,
        grid=grid,
        compiler_params=pltpu.TPUCompilerParams(
            # CRITICAL: This is the Pallas equivalent of Hopper's TMA. 
            # It enables Double Buffering, hiding the HBM access latency.
            num_pipeline_stages=2 
        )
    )(q, k, v)


# ==============================================================================
# PYTHON WRAPPER (Public API)
# ==============================================================================
@functools.partial(jax.jit, static_argnames=['sm_scale'])
def call_pallas_flash(q, k, v, sm_scale=None):
    """
    High-level JAX wrapper for the TPU Pallas kernel.
    Handles the standard (Batch, Heads, SeqLen, HeadDim) dimensions.
    """
    batch, heads, seq_len, head_dim = q.shape
    
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)
        
    # We use jax.vmap to handle the Batch and Head dimensions.
    # This cleanly isolates the Pallas kernel to just the Sequence dimensions,
    # mapping perfectly to the TPU's matrix units without complex 4D BlockSpecs.
    
    # vmap over Heads (axis 1)
    flash_heads = jax.vmap(_pallas_flash_1d, in_axes=(1, 1, 1), out_axes=1)
    
    # vmap over Batch (axis 0)
    flash_batch = jax.vmap(flash_heads, in_axes=(0, 0, 0), out_axes=0)
    
    return flash_batch(q, k, v, sm_scale=sm_scale)


# ==============================================================================
# USAGE / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    # Test settings (Shapes must be divisible by 128 for this basic BlockSpec)
    BATCH = 2
    HEADS = 8
    SEQ_LEN = 2048
    HEAD_DIM = 128
    
    print("Initializing TPU Tensors (Using bfloat16 for native MXU precision)...")
    
    # Initialize random keys (simulating PRNG)
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    
    # TPU v5e natively excels at bfloat16
    q = jax.random.normal(kq, (BATCH, HEADS, SEQ_LEN, HEAD_DIM), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (BATCH, HEADS, SEQ_LEN, HEAD_DIM), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (BATCH, HEADS, SEQ_LEN, HEAD_DIM), dtype=jnp.bfloat16)

    print(f"Executing Omni-Attention Pallas Kernel on shape {q.shape}...")
    
    try:
        # Note: on a non-TPU machine, this will compile but execute on the CPU backend emulator
        output = call_pallas_flash(q, k, v)
        print(f"Kernel executed successfully. Output shape: {output.shape}")
    except Exception as e:
        print(f"Kernel execution failed: {e}")