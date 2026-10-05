#include "gpu_compute.h"
#include <cstring>

namespace qlhs::gpu {

#define GC_CHECK(x, msg) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  err = std::string(msg) + " (VkResult " + std::to_string((int)_r) + ")"; return false; } } while (0)
#define GC_CHECKR(x, msg, ret) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  if (err) *err = std::string(msg) + " (VkResult " + std::to_string((int)_r) + ")"; return ret; } } while (0)

uint32_t GpuCompute::findMem(uint32_t bits, VkMemoryPropertyFlags want) const {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

bool GpuCompute::init(std::string &err) {
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1; app.pApplicationName = "qlhs-gpu";
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
  GC_CHECK(vkCreateInstance(&ici, nullptr, &inst_), "vkCreateInstance");

  uint32_t nd = 0; vkEnumeratePhysicalDevices(inst_, &nd, nullptr);
  if (!nd) { err = "no Vulkan physical devices"; return false; }
  std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst_, &nd, devs.data());
  for (auto d : devs) {
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, q.data());
    for (uint32_t i = 0; i < nq; i++) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { phys_ = d; qf_ = i; break; }
    if (phys_) break;
  }
  if (!phys_) { err = "no compute queue family"; return false; }
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys_, &pp);
  std::strncpy(devName_, pp.deviceName, sizeof devName_ - 1);

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qf_; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  GC_CHECK(vkCreateDevice(phys_, &dci, nullptr, &dev_), "vkCreateDevice");
  vkGetDeviceQueue(dev_, qf_, 0, &queue_);

  VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci.queueFamilyIndex = qf_; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  GC_CHECK(vkCreateCommandPool(dev_, &cpci, nullptr, &cmdPool_), "vkCreateCommandPool");

  VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256},
                                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1024}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  dpci.maxSets = 512; dpci.poolSizeCount = 2; dpci.pPoolSizes = ps;
  GC_CHECK(vkCreateDescriptorPool(dev_, &dpci, nullptr, &descPool_), "vkCreateDescriptorPool");
  return true;
}

void GpuCompute::shutdown() {
  if (dev_) vkDeviceWaitIdle(dev_);
  if (descPool_) vkDestroyDescriptorPool(dev_, descPool_, nullptr);
  if (cmdPool_) vkDestroyCommandPool(dev_, cmdPool_, nullptr);
  if (dev_) vkDestroyDevice(dev_, nullptr);
  if (inst_) vkDestroyInstance(inst_, nullptr);
  *this = GpuCompute{};
}

Buffer GpuCompute::alloc(VkDeviceSize bytes, bool uniform, std::string *err) {
  Buffer b{}; b.size = bytes;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = bytes;
  bci.usage = uniform ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  GC_CHECKR(vkCreateBuffer(dev_, &bci, nullptr, &b.buf), "vkCreateBuffer", b);
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev_, b.buf, &mr);
  uint32_t mt = findMem(mr.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  if (mt == UINT32_MAX) { if (err) *err = "no host-visible memory type"; return b; }
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
  GC_CHECKR(vkAllocateMemory(dev_, &mai, nullptr, &b.mem), "vkAllocateMemory", b);
  GC_CHECKR(vkBindBufferMemory(dev_, b.buf, b.mem, 0), "vkBindBufferMemory", b);
  GC_CHECKR(vkMapMemory(dev_, b.mem, 0, bytes, 0, &b.map), "vkMapMemory", b);
  return b;
}

void GpuCompute::free(Buffer &b) {
  if (b.map) vkUnmapMemory(dev_, b.mem);
  if (b.buf) vkDestroyBuffer(dev_, b.buf, nullptr);
  if (b.mem) vkFreeMemory(dev_, b.mem, nullptr);
  b = Buffer{};
}

Kernel GpuCompute::loadKernel(const uint32_t *spv, size_t words,
                              const std::vector<VkDescriptorType> &bindings, uint32_t localX,
                              std::string *err, const char *entry) {
  Kernel k{}; k.bindings = bindings; k.localX = localX ? localX : 1;
  std::vector<VkDescriptorSetLayoutBinding> b(bindings.size());
  for (size_t i = 0; i < bindings.size(); i++)
    b[i] = {(uint32_t)i, bindings[i], 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = (uint32_t)b.size(); dslci.pBindings = b.data();
  GC_CHECKR(vkCreateDescriptorSetLayout(dev_, &dslci, nullptr, &k.dsl), "dsl", k);
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1; plci.pSetLayouts = &k.dsl;
  GC_CHECKR(vkCreatePipelineLayout(dev_, &plci, nullptr, &k.layout), "pl", k);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = words * 4; smci.pCode = spv;
  VkShaderModule sm;
  GC_CHECKR(vkCreateShaderModule(dev_, &smci, nullptr, &sm), "shaderModule", k);
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = entry;
  cpci.layout = k.layout;
  VkResult r = vkCreateComputePipelines(dev_, VK_NULL_HANDLE, 1, &cpci, nullptr, &k.pipe);
  vkDestroyShaderModule(dev_, sm, nullptr);
  if (r != VK_SUCCESS) { if (err) *err = "vkCreateComputePipelines"; }
  return k;
}

void GpuCompute::destroyKernel(Kernel &k) {
  if (k.pipe) vkDestroyPipeline(dev_, k.pipe, nullptr);
  if (k.layout) vkDestroyPipelineLayout(dev_, k.layout, nullptr);
  if (k.dsl) vkDestroyDescriptorSetLayout(dev_, k.dsl, nullptr);
  k = Kernel{};
}

bool GpuCompute::run(const std::vector<Step> &steps, std::string *err) {
  // one descriptor set per step
  std::vector<VkDescriptorSet> sets(steps.size(), VK_NULL_HANDLE);
  for (size_t s = 0; s < steps.size(); s++) {
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool_; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &steps[s].k->dsl;
    GC_CHECKR(vkAllocateDescriptorSets(dev_, &dsai, &sets[s]), "allocDescriptorSet", false);
    const auto &st = steps[s];
    std::vector<VkDescriptorBufferInfo> dbi(st.bufs.size());
    std::vector<VkWriteDescriptorSet> w(st.bufs.size());
    for (size_t i = 0; i < st.bufs.size(); i++) {
      dbi[i] = {st.bufs[i]->buf, 0, st.bufs[i]->size};
      w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet = sets[s]; w[i].dstBinding = (uint32_t)i;
      w[i].descriptorCount = 1; w[i].descriptorType = st.k->bindings[i]; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev_, (uint32_t)w.size(), w.data(), 0, nullptr);
  }

  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cmdPool_; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; GC_CHECKR(vkAllocateCommandBuffers(dev_, &cbai, &cb), "allocCmd", false);
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  GC_CHECKR(vkBeginCommandBuffer(cb, &cbbi), "beginCmd", false);
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  for (size_t s = 0; s < steps.size(); s++) {
    const auto &st = steps[s];
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.k->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, st.k->layout, 0, 1, &sets[s], 0, nullptr);
    uint32_t groups = (st.nElems + st.k->localX - 1) / st.k->localX;
    vkCmdDispatch(cb, groups ? groups : 1, 1, 1);
    if (s + 1 < steps.size())
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr);
  }
  GC_CHECKR(vkEndCommandBuffer(cb), "endCmd", false);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence;
  GC_CHECKR(vkCreateFence(dev_, &fci, nullptr, &fence), "fence", false);
  GC_CHECKR(vkQueueSubmit(queue_, 1, &si, fence), "submit", false);
  GC_CHECKR(vkWaitForFences(dev_, 1, &fence, VK_TRUE, UINT64_MAX), "wait", false);
  vkDestroyFence(dev_, fence, nullptr);
  vkFreeCommandBuffers(dev_, cmdPool_, 1, &cb);
  if (!sets.empty()) vkFreeDescriptorSets(dev_, descPool_, (uint32_t)sets.size(), sets.data());
  return true;
}

} // namespace qlhs::gpu
