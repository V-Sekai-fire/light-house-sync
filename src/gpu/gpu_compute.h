// GpuCompute: a small headless Vulkan compute context for the QuestLHSync GPU solver.
// Owns instance/device/queue, allocates host-visible buffers, loads SPIR-V kernels with a
// fixed descriptor layout, and records chained dispatches (with compute->compute barriers)
// into one command buffer. This is the device layer the ported L-BFGS-B driver drives; it
// replaces lbfgsb's rdc::Device backend with plain Vulkan (works inside vrserver, proven).
//
// Build with: zig c++ -std=c++17
#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <vector>

namespace qlhs::gpu {

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  void *map = nullptr;
  VkDeviceSize size = 0;
};

struct Kernel {
  VkPipeline pipe = VK_NULL_HANDLE;
  VkPipelineLayout layout = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
  std::vector<VkDescriptorType> bindings;  // in binding-index order (UNIFORM_BUFFER or STORAGE_BUFFER)
  uint32_t localX = 1;                     // numthreads.x, for group-count math
};

class GpuCompute {
public:
  bool init(std::string &err);
  void shutdown();
  const char *deviceName() const { return devName_; }

  // Host-visible (HOST_VISIBLE|HOST_COHERENT) buffer; `uniform` picks UNIFORM vs STORAGE usage.
  Buffer alloc(VkDeviceSize bytes, bool uniform = false, std::string *err = nullptr);
  void free(Buffer &b);

  // Load a compute kernel from SPIR-V words with its binding types (binding-index order).
  // entry defaults to "main"; localX is the kernel's numthreads.x for dispatch math.
  Kernel loadKernel(const uint32_t *spv, size_t words,
                    const std::vector<VkDescriptorType> &bindings, uint32_t localX,
                    std::string *err = nullptr, const char *entry = "main");
  void destroyKernel(Kernel &k);

  // Record a chain of dispatches, then submit+wait. Each step binds `bufs` (in binding order)
  // to `k` and dispatches ceil(nElems/k.localX) groups; a barrier is inserted between steps.
  struct Step { const Kernel *k; std::vector<const Buffer *> bufs; uint32_t nElems; };
  bool run(const std::vector<Step> &steps, std::string *err = nullptr);

  VkDevice device() const { return dev_; }

private:
  uint32_t findMem(uint32_t bits, VkMemoryPropertyFlags want) const;

  VkInstance inst_ = VK_NULL_HANDLE;
  VkPhysicalDevice phys_ = VK_NULL_HANDLE;
  VkDevice dev_ = VK_NULL_HANDLE;
  VkQueue queue_ = VK_NULL_HANDLE;
  uint32_t qf_ = 0;
  VkCommandPool cmdPool_ = VK_NULL_HANDLE;
  VkDescriptorPool descPool_ = VK_NULL_HANDLE;
  char devName_[256] = {0};
};

} // namespace qlhs::gpu
