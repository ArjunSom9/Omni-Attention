import torch
import triton
import triton.language as tl

# ==============================================================================
# OMNI-ATTENTION: HOPPER-OPTIMIZED KERNEL (FLASH-ATTENTION 3 LOGIC)
# Architecture Target: NVIDIA H100 (Hopper)
# Core Mechanics: TMA (Tensor Memory Accelerator), WGMMA, L2 Cache Swizzling
# ==============================================================================

@triton.jit
def _fwd_kernel(
    # Pointers to matrices
    Q, K, V, sm_scale,
    Out,
    # Strides to handle memory layout
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    # Matrix dimensions
    Z, H, N_CTX,
    # Meta-parameters (Compile-time constants)
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton Kernel implementing the FlashAttention-3 forward pass pipeline.
    """
    # -------------------------------------------------------------------------
    # 1. L2 Cache Swizzling (Section 3.3 Layout Optimization)
    # -------------------------------------------------------------------------
    # We swizzle the thread block (CTA) IDs to ensure that CTAs accessing the
    # same KV blocks are spatially close, maximizing L2 cache hit rates and
    # preventing partition camping.
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    num_m_blocks = tl.cdiv(N_CTX, BLOCK_M)
    
    # Swizzle logic: group blocks in chunks to increase locality
    GROUP_M = 8 
    group_id = start_m // GROUP_M
    group_size = min(num_m_blocks - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (start_m % group_size)
    
    off_z = off_hz // H
    off_h = off_hz % H

    # -------------------------------------------------------------------------
    # 2. TMA Block Pointers (Section 3.1 The Triton Programming Model)
    # -------------------------------------------------------------------------
    # tl.make_block_ptr creates a 2D descriptor for the TMA. On Hopper, this
    # compiles down to `cp.async.bulk.tensor` instructions, offloading memory
    # address calculations and boundary checks to hardware.
    
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh

    # Q Block Pointer: Shape (N_CTX, BLOCK_DMODEL), Block (BLOCK_M, BLOCK_DMODEL)
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # K Block Pointer: Shape (BLOCK_DMODEL, N_CTX), Block (BLOCK_DMODEL, BLOCK_N)
    # Note: K is transposed in the block descriptor for WGMMA dot product
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    
    # V Block Pointer: Shape (N_CTX, BLOCK_DMODEL), Block (BLOCK_N, BLOCK_DMODEL)
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # -------------------------------------------------------------------------
    # 3. Prologue: Initialize State
    # -------------------------------------------------------------------------
    # Online Softmax requires maintaining the max (m_i) and the sum (l_i)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load Q tile
    # On Hopper, Q stays in registers/shared memory for the entire inner loop
    q = tl.load(Q_block_ptr)

    # -------------------------------------------------------------------------
    # 4. Asynchronous Pipeline Design (Section 3.2.1)
    # -------------------------------------------------------------------------
    # Note: Modern Triton handles the explicit loop pipelining (loading `next`
    # while computing `current`) automatically when using `num_stages > 1`.
    # The compiler emits the asynchronous TMA loads and overlaps them with 
    # WGMMA instructions.
    
    for start_n in range(0, N_CTX, BLOCK_N):
        # 4a. Load K and V tiles via TMA
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # 4b. Compute S = Q * K^T (WGMMA instruction emitted here)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * sm_scale

        # 4c. Compute Softmax (ALU operation)
        # Interleaved with WGMMA latency by the compiler
        m_ij = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # Online softmax math
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        # 4d. Update running sum
        l_i_new = alpha * l_i + tl.sum(p, 1)
        
        # 4e. Scale accumulated output and add new V
        # Compute O = P * V (WGMMA instruction emitted here)
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        
        # Cast P to fp16/bf16 to utilize tensor cores for the second dot product
        p = p.to(v.dtype)
        acc += tl.dot(p, v) / l_i_new[:, None]

        # 4f. Pipeline Advance
        m_i = m_i_new
        l_i = l_i_new
        
        # Advance TMA pointers to the next block along the sequence dimension
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # -------------------------------------------------------------------------
    # 5. Epilogue: Write Output
    # -------------------------------------------------------------------------
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    
    # Cast accumulator back to input dtype and write asynchronously via TMA
    tl.store(O_block_ptr, acc.to(Out.dtype))


# ==============================================================================
# PYTHON WRAPPER
# ==============================================================================
def flash_attn_v3_forward(q, k, v, sm_scale):
    """
    Python wrapper for launching the Triton FlashAttention-3 kernel.
    Args:
        q, k, v: Tensors of shape (Batch, Heads, SeqLen, HeadDim)
        sm_scale: Scaling factor (usually 1 / sqrt(HeadDim))
    Returns:
        out: Tensor of shape (Batch, Heads, SeqLen, HeadDim)
    """
    # Shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk == Lv, "Query, Key, and Value must have the same Head Dimension"
    assert Lq in {16, 32, 64, 128, 256}, "Head dimension must be a power of 2 up to 256"
    
    Z, H, N_CTX, D_HEAD = q.shape

    # Allocate output tensor
    out = torch.empty_like(q)

    # -------------------------------------------------------------------------
    # Tuning Parameters (Targeted for H100 WGMMA)
    # -------------------------------------------------------------------------
    # BLOCK_M = 128 aligns with the 128-thread warpgroup size on Hopper
    BLOCK_M = 128
    BLOCK_N = 128 if D_HEAD <= 128 else 64
    
    # num_stages triggers Triton's software pipelining capability, allowing
    # multi-buffering for the TMA to hide global memory latency.
    # num_warps=8 allows 2 warpgroups (256 threads) per CTA.
    num_stages = 4 if D_HEAD <= 128 else 3
    num_warps = 8 

    # Grid dimensions: (SeqLen / BLOCK_M, Batch * Heads, 1)
    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)

    # Launch kernel
    _fwd_kernel[grid](
        q, k, v, sm_scale,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        BLOCK_M=BLOCK_M, BLOCK_DMODEL=D_HEAD, BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages
    )

    return out


# ==============================================================================
# USAGE / VERIFICATION
# ==============================================================================
if __name__ == "__main__":
    # Test settings
    BATCH = 2
    HEADS = 8
    SEQ_LEN = 4096
    HEAD_DIM = 128
    
    # Ensure standard contiguous memory layouts for testing
    q = torch.randn((BATCH, HEADS, SEQ_LEN, HEAD_DIM), device='cuda', dtype=torch.float16)
    k = torch.randn((BATCH, HEADS, SEQ_LEN, HEAD_DIM), device='cuda', dtype=torch.float16)
    v = torch.randn((BATCH, HEADS, SEQ_LEN, HEAD_DIM), device='cuda', dtype=torch.float16)
    
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    print(f"Executing Omni-Attention Hopper Kernel...")
    print(f"Configuration: Batch={BATCH}, Heads={HEADS}, Seq={SEQ_LEN}, Dim={HEAD_DIM}")
    
    try:
        # Run Kernel
        output = flash_attn_v3_forward(q, k, v, sm_scale)
        print(f"Kernel executed successfully. Output shape: {output.shape}")
        
    except Exception as e:
        print(f"Kernel launch failed (Ensure you are on an NVIDIA GPU with Triton installed): {e}")