#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "llm/cuda_check.h"

namespace llm {

// Owning RAII wrapper around one cudaMalloc.
//
// Deliberately dumb: one allocation, one free, no pooling. The custom
// allocator (bump for weights / paged for KV / arena for activations) replaces
// every use of this later; keeping the ad-hoc allocations behind a single type
// now means that swap is a change of type, not a hunt for cudaFree calls.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t bytes) { allocate(bytes); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), bytes_(other.bytes_) {
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      bytes_ = other.bytes_;
      other.ptr_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { reset(); }

  void allocate(std::size_t bytes) {
    reset();
    if (bytes == 0) return;
    LLM_CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    bytes_ = bytes;
  }

  void reset() {
    if (ptr_ != nullptr) {
      // Destructor path: a failure here can't be propagated, and cudaFree on a
      // valid pointer only fails if the context is already gone.
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    bytes_ = 0;
  }

  std::size_t bytes() const { return bytes_; }

  template <typename T>
  T* as() {
    return static_cast<T*>(ptr_);
  }

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(ptr_);
  }

  void upload(const void* host, std::size_t bytes) {
    LLM_CUDA_CHECK(cudaMemcpy(ptr_, host, bytes, cudaMemcpyHostToDevice));
  }

  void download(void* host, std::size_t bytes) const {
    LLM_CUDA_CHECK(cudaMemcpy(host, ptr_, bytes, cudaMemcpyDeviceToHost));
  }

  void zero() {
    if (bytes_ != 0) LLM_CUDA_CHECK(cudaMemset(ptr_, 0, bytes_));
  }

 private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

// Convenience: upload a host vector into a freshly sized buffer.
template <typename T>
DeviceBuffer make_device_buffer(const std::vector<T>& host) {
  DeviceBuffer buf(host.size() * sizeof(T));
  buf.upload(host.data(), host.size() * sizeof(T));
  return buf;
}

}  // namespace llm
