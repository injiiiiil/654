#pragma once

#include <ATen/native/vulkan/api/Common.h>
#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Command.h>
#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Pipeline.h>
#include <ATen/native/vulkan/api/Resource.h>
#include <ATen/native/vulkan/api/Shader.h>

namespace at {
namespace native {
namespace vulkan {
namespace api {

//
// Vulkan Context holds onto all relevant Vulkan state as it pertains to our
// use of Vulkan in PyTorch.  A Context is associated with one, and only one,
// Adapter as a precursor to multi-GPU support.  All Vulkan tensors in PyTorch
// are associated with a Context to make tensor <-> device affinity explicit.
// The context is currently a global object, but technically it does not need
// to be if we were to make it explicit to the user.
//

class Context final {
 public:
  explicit Context(const Adapter& adapter);
  Context(const Context&) = delete;
  Context(Context&&) = delete;
  Context& operator=(const Context&) = delete;
  Context& operator=(Context&&) = delete;
  ~Context();

  GPU gpu();
  Command& command();
  Shader& shader();
  Pipeline& pipeline();
  Descriptor& descriptor();
  Resource& resource();

  void dispatch(
      Command::Buffer& command_buffer,
      const Shader::Layout::Descriptor& shader_layout_descriptor,
      const Shader::Descriptor& shader_descriptor,
      const Descriptor::Set& descriptor_set);

  // This function is very expensive and its use is bad for performance.
  // Only use this function for debugging or as a short term hack on way to a
  // more performant solution.

  void flush();

 private:
  VkDevice device();
  VkQueue queue();

 private:
  class Deleter final {
   public:
    void operator()(VkDevice device) const;
  };

 private:
  // Construction and destruction order matters.  Do not move members around.
  Adapter adapter_;
  Handle<VkDevice, decltype(&VK_DELETER(Device))> device_;
  VkQueue queue_;
  Command command_;
  Shader shader_;
  Pipeline pipeline_;
  Descriptor descriptor_;
  Resource resource_;
};

Context* context();

//
// Impl
//

inline GPU Context::gpu() {
  // A GPU is simply a (physical device, logical device, device queue) trio.
  return {
    &adapter_,
    device(),
    queue(),
  };
}

inline Command& Context::command() {
  return command_;
}

inline Shader& Context::shader() {
  return shader_;
}

inline Pipeline& Context::pipeline() {
  return pipeline_;
}

inline Descriptor& Context::descriptor() {
  return descriptor_;
}

inline Resource& Context::resource() {
  return resource_;
}

inline VkDevice Context::device() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(device_);
  return device_.get();
}

inline VkQueue Context::queue() {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(queue_);
  return queue_;
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
