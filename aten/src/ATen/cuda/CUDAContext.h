#pragma once

#include <ATen/core/ATenGeneral.h>
#include <ATen/Context.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cstdint>

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>

namespace at {
namespace cuda {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

struct CAFFE2_API CUDAP2PState {
  CUDAP2PState(int64_t src, int64_t target) 
  : src_device(src), target_device(target) {
    auto num_gpus = c10::cuda::device_count();
    AT_ASSERT(src_device >= 0 && src_device < num_gpus);
    AT_ASSERT(target_device >= 0 && target_device < num_gpus);
    // if same device, no need for enabling
    if(src_device == target_device) {
      p2p_enable_status = true;
    // if existing connection there, no need for enabling
    } else if (!(p2p_enable_status)) {
      auto current_device = c10::cuda::current_device();
      c10::cuda::set_device(src_device);

      int canAccess;
      AT_CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, src_device, target_device));
      if (canAccess) {
        cudaError_t err = cudaDeviceEnablePeerAccess(target_device, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled || err == cudaSuccess) {
          p2p_enable_status = true;
        } else {
          // if a device has more than 8 connections simultaneously (current max limit), we should
          // get a cudaErrorTooManyPeers
          p2p_enable_status = false;
          std::string message = "P2P connection could not be established. Got cudaError_t: ";
          message + std::string(cudaGetErrorName(err));
          AT_WARN(message);
        }
        cudaGetLastError();
      } else {
        p2p_enable_status = false;
        AT_WARN("The src device and the target device does not support P2P.");
      }
      c10::cuda::set_device(current_device);
    }
  }

  ~CUDAP2PState() {
    // when done with the connection, make sure to release the connection
    // since currently there is a max limit of 8 devices that can establish
    // p2p connections with another device, simultaneously
    if (p2p_enable_status && src_device != target_device) {
      auto current_device = c10::cuda::current_device();
      c10::cuda::set_device(src_device);
      cudaDeviceDisablePeerAccess(target_device);
      c10::cuda::set_device(current_device);
    }
  }

  bool isEnabled() {
    return p2p_enable_status;
  }

  private:
    bool p2p_enable_status = false;
    int64_t src_device;
    int64_t target_device;
};

/* Device info */
inline int64_t getNumGPUs() {
    return c10::cuda::device_count();
}

/**
 * In some situations, you may have compiled with CUDA, but no CUDA
 * device is actually available.  Test for this case using is_available().
 */
inline bool is_available() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err == cudaErrorInsufficientDriver) {
      return false;
    }
    return count > 0;
}

CAFFE2_API cudaDeviceProp* getCurrentDeviceProperties();

CAFFE2_API int warp_size();

CAFFE2_API cudaDeviceProp* getDeviceProperties(int64_t device);

CAFFE2_API Allocator* getCUDADeviceAllocator();

/* Handles */
CAFFE2_API cusparseHandle_t getCurrentCUDASparseHandle();
CAFFE2_API cublasHandle_t getCurrentCUDABlasHandle();


} // namespace cuda
} // namespace at
