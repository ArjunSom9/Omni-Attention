#pragma once

// ==============================================================================
// OMNI-ATTENTION: OMNI-SERVE INFERENCE RUNTIME (HEADER)
// Project Phase 4: Production C++ Serving Infrastructure
// Architecture Targets: C++ PJRT API (Pre-JIT Runtime) for TPU/GPU
// ==============================================================================

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <cstdint>

// Conceptual XLA/PJRT Headers (Provided by TensorFlow/XLA Bazel toolchain)
namespace xla {
    class PjRtClient;
    class PjRtBuffer;
    class PjRtLoadedExecutable;
}

namespace omni_serve {

// ------------------------------------------------------------------------------
// 1. DATA STRUCTURES
// ------------------------------------------------------------------------------

/**
 * @brief Represents a single LLM inference request.
 * Required for Continuous Batching where requests arrive asynchronously.
 */
struct InferenceRequest {
    uint64_t request_id;
    std::vector<int32_t> prompt_token_ids;
    size_t max_new_tokens;
    float temperature;
    float top_p;
    
    // Internal state tracking
    bool is_prefill_complete = false;
    std::vector<int32_t> generated_tokens;
};

/**
 * @brief A lightweight container for mapping host memory to device memory.
 */
struct TensorData {
    void* host_ptr;
    std::vector<int64_t> dimensions;
    size_t byte_size;
    // e.g., fp16, bf16, fp8_e4m3fn
    std::string dtype; 
};

// ------------------------------------------------------------------------------
// 2. THE PJRT BACKEND (Section 8.2: Implementing the PJRT Client)
// ------------------------------------------------------------------------------

/**
 * @brief The core execution engine. Bypasses Python entirely.
 * Loads StableHLO IR directly into the device compiler and manages
 * DMA transfers from Host (CPU) to Device (HBM) using the PJRT API.
 */
class OmniServeBackend {
public:
    OmniServeBackend() = default;
    ~OmniServeBackend() = default;

    /**
     * @brief Initializes the underlying hardware client.
     * @param device_type "TPU" (loads tpu_client) or "GPU" (loads gpu_client).
     */
    void Initialize(const std::string& device_type);

    /**
     * @brief Compiles a serialized StableHLO program (e.g., our Pallas/Triton kernels)
     * into a device-specific executable.
     * @param hlo_proto_path Path to the exported .mlir or .pb HLO file.
     */
    void LoadModel(const std::string& hlo_proto_path);

    /**
     * @brief The ultra-low-latency execution path. 
     * Takes raw Host pointers, DMA copies them to HBM, and triggers the hardware kernels.
     * @param input_data Vector of host tensors.
     * @return Vector of device buffers containing the result (e.g., logits).
     */
    std::vector<std::shared_ptr<xla::PjRtBuffer>> Execute(const std::vector<TensorData>& input_data);

    /**
     * @brief Transfers device memory back to host memory (blocking).
     */
    void TransferToHost(std::shared_ptr<xla::PjRtBuffer> device_buffer, void* host_dst);

private:
    // The unified accelerator interface
    std::shared_ptr<xla::PjRtClient> client_;
    
    // The compiled StableHLO kernels residing on the device
    std::vector<std::unique_ptr<xla::PjRtLoadedExecutable>> executables_;
};

// ------------------------------------------------------------------------------
// 3. CONTINUOUS BATCHING SCHEDULER (Section 8.3)
// ------------------------------------------------------------------------------

/**
 * @brief Manages asynchronous incoming RPCs and maps logical tokens to physical memory.
 * This resolves the inefficiency of waiting for a whole batch to finish before
 * inserting new requests.
 */
class RequestScheduler {
public:
    RequestScheduler(size_t max_physical_blocks, size_t block_size);
    ~RequestScheduler();

    /**
     * @brief Thread-safe ingestion of new generation requests.
     */
    void AddRequest(const InferenceRequest& req);

    /**
     * @brief Formats the next physical batch for the PJRT Backend to execute.
     * Performs Block Table allocation for new tokens.
     */
    std::vector<InferenceRequest> StepAndGetNextBatch();

private:
    // Concurrency control for incoming gRPC/REST requests
    std::mutex queue_mutex_;
    std::queue<InferenceRequest> pending_requests_;
    std::vector<InferenceRequest> active_requests_;

    // Paged Attention Block Table Logic
    size_t block_size_;
    size_t max_blocks_;
    std::vector<bool> free_blocks_; // Tracks physical memory availability in KV cache
    
    // Maps Request ID -> Vector of Physical Block Indices in the HBM KV Cache
    std::unordered_map<uint64_t, std::vector<size_t>> block_table_;

    // Internal helper to allocate a new physical block to a request
    bool AllocateBlock(uint64_t request_id);
};

// ------------------------------------------------------------------------------
// 4. XLA CUSTOM CALLS (Section 8.4)
// ------------------------------------------------------------------------------

/**
 * @brief C++ operations that augment the XLA graph.
 * Used for operations that are highly irregular and poorly suited for standard HLO.
 */
namespace CustomOps {

    /**
     * @brief Specialized Token Sampling (Top-P / Nucleus Sampling)
     * Directly manipulates raw memory pointers conforming to the XLA FFI ABI.
     * * @param out Pointer to the output buffer (the sampled token ID).
     * @param in Array of pointers to input buffers (logits, top_p scalar, random seed).
     */
    extern "C" void CustomTopP(void* out, void** in);

    /**
     * @brief Registers the C++ functions with the XLA compiler so they can be
     * resolved during the Execute() call.
     */
    void RegisterCustomCalls();

} // namespace CustomOps

} // namespace omni_serve