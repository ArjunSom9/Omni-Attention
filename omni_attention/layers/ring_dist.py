import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
import functools

# ==============================================================================
# OMNI-ATTENTION: DISTRIBUTED RING ATTENTION
# Architecture Target: TPU Pods / Multi-GPU Clusters (via XLA SPMD)
# Core Mechanics: shard_map, lax.ppermute, Computation-Communication Overlap
# ==============================================================================

def make_ring_attention_sharded(ring_size: int, axis_name: str = 'ring'):
    """
    Creates a shard_map-compatible Ring Attention function.
    We use a closure to statically generate the permutation list required by XLA.
    """
    # -------------------------------------------------------------------------
    # 1. Define the Topology Rotation (Section 5.2.2)
    # -------------------------------------------------------------------------
    # For Ring Attention, Device i sends its K,V blocks to Device (i+1) % N.
    # This maps optimally to the 3D Torus interconnect of TPU architectures.
    ring_perm = [(i, (i + 1) % ring_size) for i in range(ring_size)]

    def _ring_attention_spmd(q_local, k_local, v_local, sm_scale):
        """
        Inner SPMD function. Runs independently on every device.
        q_local, k_local, v_local are the shards of the sequence residing in 
        this specific device's HBM.
        """
        batch, heads, seq_len, head_dim = q_local.shape

        # Initialize running statistics for Online Softmax (FlashAttention math)
        # Because we are accumulating across multiple discrete K,V blocks arriving
        # over the network, we MUST use the stable associative softmax reductions.
        acc = jnp.zeros_like(q_local)
        m_i = jnp.full((batch, heads, seq_len, 1), -jnp.inf, dtype=jnp.float32)
        l_i = jnp.zeros((batch, heads, seq_len, 1), dtype=jnp.float32)

        # The rotating buffers
        k_curr, v_curr = k_local, v_local

        # ---------------------------------------------------------------------
        # 2. Computation-Communication Overlap Loop (Section 5.2.3)
        # ---------------------------------------------------------------------
        for step in range(ring_size):
            # A) START ASYNC NETWORK TRANSFER (DMA)
            # We trigger the ppermute for the *next* iteration immediately.
            # By not consuming k_next/v_next until the loop repeats, XLA knows 
            # it can schedule this network transfer concurrently with the math below.
            if step < ring_size - 1:
                k_next = jax.lax.ppermute(k_curr, axis_name=axis_name, perm=ring_perm)
                v_next = jax.lax.ppermute(v_curr, axis_name=axis_name, perm=ring_perm)

            # B) LOCAL COMPUTE (MXU / Tensor Cores)
            # Compute S = Q * K^T
            qk = jnp.einsum('b h s d, b h k d -> b h s k', q_local, k_curr) * sm_scale

            # Online Softmax updates
            m_ij = jnp.max(qk, axis=-1, keepdims=True)
            m_new = jnp.maximum(m_i, m_ij)

            alpha = jnp.exp(m_i - m_new)
            p = jnp.exp(qk - m_new)

            l_new = alpha * l_i + jnp.sum(p, axis=-1, keepdims=True)

            # Compute O = P * V and accumulate
            acc = acc * alpha + jnp.einsum('b h s k, b h k d -> b h s d', p, v_curr)

            # Update running stats
            m_i = m_new
            l_i = l_new

            # C) ADVANCE PIPELINE
            # Move the received network buffers into the current working buffers
            if step < ring_size - 1:
                k_curr = k_next
                v_curr = v_next

        # Final scaling by the complete denominator
        out_local = acc / l_i
        return out_local

    return _ring_attention_spmd


# ==============================================================================
# PYTHON WRAPPER (Public API)
# ==============================================================================
def ring_attention(q, k, v, mesh: Mesh, sm_scale: float):
    """
    High-level API that wraps the SPMD function in a `shard_map`.
    
    Args:
        q, k, v: Full global tensors of shape (Batch, Heads, Global_Seq, HeadDim)
        mesh: jax.sharding.Mesh defining the physical device topology
        sm_scale: Softmax scaling factor
    """
    # Extract the number of devices participating in the sequence (ring) dimension
    ring_size = mesh.shape['ring']
    
    # Create the specialized SPMD function
    spmd_fn = make_ring_attention_sharded(ring_size=ring_size, axis_name='ring')
    
    # Wrap in shard_map.
    # PartitionSpec(None, None, 'ring', None) means:
    # - Batch (None) is not sharded
    # - Heads (None) is not sharded
    # - Sequence ('ring') IS sharded across the ring axis of the mesh
    # - HeadDim (None) is not sharded
    sharded_fn = shard_map(
        spmd_fn,
        mesh=mesh,
        in_specs=(
            P(None, None, 'ring', None), # Q
            P(None, None, 'ring', None), # K
            P(None, None, 'ring', None), # V
            P()                          # sm_scale (scalar)
        ),
        out_specs=P(None, None, 'ring', None), # Output is also sharded along seq
        check_rep=False
    )
    
    return sharded_fn(q, k, v, sm_scale)


# ==============================================================================
# USAGE / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    import os
    
    # -------------------------------------------------------------------------
    # Setup Simulated Devices (For local verification without a TPU Pod)
    # -------------------------------------------------------------------------
    # Force JAX to emulate 4 distinct hardware devices on the CPU backend
    os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'
    devices = jax.devices()
    print(f"Detected {len(devices)} simulated JAX devices: {devices}\n")
    
    # Define our hyper-parameters
    BATCH = 1
    HEADS = 2
    GLOBAL_SEQ = 4096  # This will be sharded into 1024 chunks per device
    HEAD_DIM = 64
    SM_SCALE = 1.0 / (HEAD_DIM ** 0.5)

    # 1. Define the Mesh (Section 5.2.1)
    # We create a 1D mesh named 'ring' containing our 4 devices
    import numpy as np
    device_mesh = np.array(devices).reshape(4)
    mesh = Mesh(device_mesh, ('ring',))
    
    print(f"Created Mesh: {mesh}")
    print(f"Global Sequence Length ({GLOBAL_SEQ}) will be divided across {len(devices)} devices.")
    print(f"Each device will hold a Sequence length of {GLOBAL_SEQ // len(devices)} in HBM.\n")

    # 2. Create Global Data
    key = jax.random.PRNGKey(42)
    q_global = jax.random.normal(key, (BATCH, HEADS, GLOBAL_SEQ, HEAD_DIM))
    k_global = jax.random.normal(key, (BATCH, HEADS, GLOBAL_SEQ, HEAD_DIM))
    v_global = jax.random.normal(key, (BATCH, HEADS, GLOBAL_SEQ, HEAD_DIM))

    # 3. Execute Ring Attention
    print("Executing Distributed Ring Attention (shard_map + ppermute)...")
    try:
        # jax.jit will trace the shard_map, compile the ppermute overlapping 
        # instructions via XLA, and dispatch it to the emulated devices.
        sharded_attn_fn = jax.jit(functools.partial(ring_attention, mesh=mesh, sm_scale=SM_SCALE))
        
        output = sharded_attn_fn(q_global, k_global, v_global)
        
        print(f"Ring Attention completed successfully!")
        print(f"Output Shape: {output.shape}") # Should match (BATCH, HEADS, GLOBAL_SEQ, HEAD_DIM)
        
    except Exception as e:
        print(f"Distributed execution failed: {e}")