#include "../include/omni_serve.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Conceptual XLA Headers (Based on standard TensorFlow/XLA Bazel toolchain)
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tpu_client.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/shape_util.h"

// ==============================================================================
// OMNI-ATTENTION: OMNI-SERVE INFERENCE RUNTIME (PJRT CLIENT)
// Project Phase 4: Production C++ Serving Infrastructure
// Architecture Targets: C++ PJRT API for TPU/GPU Execution
// ==============================================================================

namespace omni_serve {

// Helper function to map string dtypes to XLA primitive types
xla::PrimitiveType StringToXlaPrimitive(const std::string& dtype) {
    if (dtype == "fp32") return xla::PrimitiveType::F32;
    if (dtype == "fp16") return xla::PrimitiveType::F16;
    if (dtype == "bf16") return xla::PrimitiveType::BF16;
    if (dtype == "fp8_e4m3fn") return xla::PrimitiveType::F8E4M3FN;
    if (dtype == "fp8_e5m2") return xla::PrimitiveType::F8E5M2;
    if (dtype == "int32") return xla::PrimitiveType::S32;
    throw std::invalid_argument("Unsupported dtype: " + dtype);
}

// ------------------------------------------------------------------------------
// 1. Hardware Initialization (TPU vs GPU)
// ------------------------------------------------------------------------------
void OmniServeBackend::Initialize(const std::string& device_type) {
    std::cout << "[Omni-Serve] Initializing PJRT Backend for target: " << device_type << std::endl;

    if (device_type == "TPU") {
        // Initializes the TPU Topology (e.g., ICI Torus for Google v5e)
        // Automatically discovers the local TPU chips and creates a mesh client.
        auto statusor_client = xla::GetTpuClient();
        if (!statusor_client.ok()) {
            throw std::runtime_error("Failed to initialize TPU Client: " + statusor_client.status().ToString());
        }
        client_ = std::move(statusor_client.value());
    } 
    else if (device_type == "GPU") {
        // Initializes the CUDA driver, binds to the NVIDIA Hopper architecture
        auto statusor_client = xla::GetGpuClient();
        if (!statusor_client.ok()) {
            throw std::runtime_error("Failed to initialize GPU Client: " + statusor_client.status().ToString());
        }
        client_ = std::move(statusor_client.value());
    } 
    else {
        throw std::invalid_argument("Unknown device type. Use 'TPU' or 'GPU'.");
    }

    std::cout << "[Omni-Serve] Successfully connected to " 
              << client_->device_count() << " " << device_type << " device(s)." << std::endl;
}

// ------------------------------------------------------------------------------
// 2. Load the Compiled StableHLO (From Python Pallas/Triton phase)
// ------------------------------------------------------------------------------
void OmniServeBackend::LoadModel(const std::string& hlo_proto_path) {
    std::cout << "[Omni-Serve] Loading StableHLO executable: " << hlo_proto_path << std::endl;

    // Read the serialized HLO (MLIR or Protocol Buffer) exported from JAX/PyTorch
    std::ifstream file(hlo_proto_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open HLO file at " + hlo_proto_path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string serialized_hlo = ss.str();

    // Compile the HLO into an executable binary resident on the hardware.
    // This is equivalent to PyTorch/JAX's AOT (Ahead-of-Time) compilation.
    xla::CompileOptions options;
    auto statusor_executable = client_->Compile(serialized_hlo, options);
    
    if (!statusor_executable.ok()) {
        throw std::runtime_error("Failed to compile StableHLO: " + statusor_executable.status().ToString());
    }

    executables_.push_back(std::move(statusor_executable.value()));
    std::cout << "[Omni-Serve] Executable loaded and resident on device memory." << std::endl;
}

// ------------------------------------------------------------------------------
// 3. High-Performance Execution Pipeline
// ------------------------------------------------------------------------------
std::vector<std::shared_ptr<xla::PjRtBuffer>> OmniServeBackend::Execute(
    const std::vector<TensorData>& input_data) {
    
    if (executables_.empty()) {
        throw std::runtime_error("No executable loaded! Call LoadModel() first.");
    }

    // A. Move data from Host (CPU RAM) to Device (HBM) using PJRT
    std::vector<std::unique_ptr<xla::PjRtBuffer>> device_buffers_unique;
    
    for (const auto& tensor : input_data) {
        xla::PrimitiveType element_type = StringToXlaPrimitive(tensor.dtype);
        xla::Shape shape = xla::ShapeUtil::MakeShape(element_type, tensor.dimensions);

        // Perform the asynchronous DMA transfer
        auto statusor_buffer = client_->BufferFromHostBuffer(
            tensor.host_ptr, 
            element_type, 
            tensor.dimensions, 
            /*byte_strides=*/std::nullopt, 
            xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall, 
            /*on_done_with_host_buffer=*/nullptr, 
            client_->devices()[0] // Default to Device 0 for single-chip execution
        );

        if (!statusor_buffer.ok()) {
            throw std::runtime_error("Host to Device DMA failed: " + statusor_buffer.status().ToString());
        }
        device_buffers_unique.push_back(std::move(statusor_buffer.value()));
    }

    // Convert to vector of raw pointers expected by the Execute API
    std::vector<xla::PjRtBuffer*> args;
    for (const auto& buf : device_buffers_unique) {
        args.push_back(buf.get());
    }

    // B. Execute the XLA/Pallas/Triton Kernel natively
    // Note: Execute() returns a nested vector [devices][outputs]
    xla::ExecuteOptions exec_options;
    auto statusor_results = executables_[0]->Execute({args}, exec_options);
    
    if (!statusor_results.ok()) {
        throw std::runtime_error("Device execution failed: " + statusor_results.status().ToString());
    }

    // Extract the outputs for Device 0
    auto raw_results = std::move(statusor_results.value()[0]);
    
    // C. Wrap the resulting raw output pointers in shared_ptrs to manage lifecycle
    std::vector<std::shared_ptr<xla::PjRtBuffer>> result_buffers;
    for (auto& raw_result : raw_results) {
        result_buffers.push_back(std::shared_ptr<xla::PjRtBuffer>(raw_result.release()));
    }

    return result_buffers;
}

// ------------------------------------------------------------------------------
// 4. Device to Host Transfer
// ------------------------------------------------------------------------------
void OmniServeBackend::TransferToHost(
    std::shared_ptr<xla::PjRtBuffer> device_buffer, void* host_dst) {
    
    // Block the CPU thread until the asynchronous hardware kernel finishes
    // and the DMA engine successfully copies the requested buffer back to host RAM.
    auto status = device_buffer->ToHostBuffer(host_dst);
    
    if (!status.ok()) {
        throw std::runtime_error("Device to Host DMA failed: " + status.ToString());
    }
}

} // namespace omni_serve