#include "../include/omni_serve.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// ==============================================================================
// OMNI-ATTENTION: OMNI-SERVE INFERENCE RUNTIME (SCHEDULER)
// Project Phase 4: Production C++ Serving Infrastructure
// Architecture Targets: Continuous Batching & Paged Attention Memory Management
// ==============================================================================

namespace omni_serve {

// ------------------------------------------------------------------------------
// 1. Initialization
// ------------------------------------------------------------------------------
RequestScheduler::RequestScheduler(size_t max_physical_blocks, size_t block_size)
    : block_size_(block_size), max_blocks_(max_physical_blocks) {
    
    // Initialize the physical KV cache memory pool.
    // True = available, False = in use.
    free_blocks_.resize(max_blocks_, true);
    
    std::cout << "[Omni-Serve] Scheduler initialized.\n"
              << "             Physical KV Cache Blocks: " << max_blocks_ << "\n"
              << "             Tokens per Block:         " << block_size_ << std::endl;
}

RequestScheduler::~RequestScheduler() {
    // Standard cleanup
}

// ------------------------------------------------------------------------------
// 2. Thread-Safe Ingestion (RPC Handler)
// ------------------------------------------------------------------------------
void RequestScheduler::AddRequest(const InferenceRequest& req) {
    // This lock protects the queue from asynchronous incoming gRPC/REST threads
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    pending_requests_.push(req);
    std::cout << "[Omni-Serve] Ingested Request ID: " << req.request_id 
              << " | Prompt Length: " << req.prompt_token_ids.size() << std::endl;
}

// ------------------------------------------------------------------------------
// 3. Physical Memory Allocation
// ------------------------------------------------------------------------------
bool RequestScheduler::AllocateBlock(uint64_t request_id) {
    // Linear scan for the first available physical block in the HBM array.
    // In a highly optimized system, this would use a bitmap/free-list.
    for (size_t i = 0; i < max_blocks_; ++i) {
        if (free_blocks_[i]) {
            free_blocks_[i] = false; // Mark as mapped
            block_table_[request_id].push_back(i); // Update logical-to-physical map
            return true;
        }
    }
    // KV Cache is full
    return false; 
}

// ------------------------------------------------------------------------------
// 4. The Continuous Batching Step (Iteration Loop)
// ------------------------------------------------------------------------------
std::vector<InferenceRequest> RequestScheduler::StepAndGetNextBatch() {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    std::vector<InferenceRequest> next_execution_batch;

    // A. PREFILL PHASE: Admit new requests if we have enough physical memory
    while (!pending_requests_.empty()) {
        const InferenceRequest& req = pending_requests_.front();
        
        // Calculate how many blocks the prompt needs initially
        size_t tokens_needed = req.prompt_token_ids.size();
        size_t blocks_needed = static_cast<size_t>(std::ceil(static_cast<float>(tokens_needed) / block_size_));
        if (blocks_needed == 0) blocks_needed = 1; // Minimum 1 block

        // Count available memory
        size_t available_blocks = std::count(free_blocks_.begin(), free_blocks_.end(), true);

        if (available_blocks >= blocks_needed) {
            // We have space! Map the logical prompt to physical blocks.
            for (size_t i = 0; i < blocks_needed; ++i) {
                AllocateBlock(req.request_id);
            }
            
            active_requests_.push_back(req);
            pending_requests_.pop();
            
            std::cout << "[Omni-Serve] Admitted Request ID: " << req.request_id 
                      << " (Allocated " << blocks_needed << " blocks for prefill)." << std::endl;
        } else {
            // Hardware KV cache is saturated. 
            // We must wait for active requests to finish and free their blocks.
            break;
        }
    }

    // B. DECODE PHASE: Manage currently active generation requests
    for (auto it = active_requests_.begin(); it != active_requests_.end(); ) {
        InferenceRequest& req = *it;

        // Check if the request has reached its generation limit
        if (req.generated_tokens.size() >= req.max_new_tokens) {
            
            // 1. Free physical HBM memory back to the pool
            for (size_t block_idx : block_table_[req.request_id]) {
                free_blocks_[block_idx] = true;
            }
            
            // 2. Remove the logical mapping
            block_table_.erase(req.request_id);
            
            std::cout << "[Omni-Serve] Request ID: " << req.request_id 
                      << " completed. KV Cache blocks freed." << std::endl;
            
            // 3. Remove from active execution pool
            it = active_requests_.erase(it);
            continue;
        }

        // C. DYNAMIC ALLOCATION: Paged Attention Mapping
        // Check if the current logical sequence length is about to overflow its 
        // currently allocated physical blocks.
        size_t current_seq_len = req.prompt_token_ids.size() + req.generated_tokens.size();
        
        if (current_seq_len > 0 && current_seq_len % block_size_ == 0) {
            // We crossed a block boundary. We need 1 more physical block to continue.
            if (!AllocateBlock(req.request_id)) {
                
                std::cout << "[Omni-Serve] WARNING: OOM for Request ID " << req.request_id 
                          << ". Preemption required!" << std::endl;
                
                // Advanced Systems (vLLM): Preempt this sequence, swap its KV cache 
                // out to Host CPU memory, and pause it. 
                // For this blueprint, we simply skip adding it to the execution batch.
                ++it;
                continue; 
            } else {
                std::cout << "[Omni-Serve] Dynamically allocated new block for Request ID: " 
                          << req.request_id << "." << std::endl;
            }
        }

        // D. Add to the batch for hardware execution
        next_execution_batch.push_back(req);
        ++it;
    }

    return next_execution_batch;
}

} // namespace omni_serve