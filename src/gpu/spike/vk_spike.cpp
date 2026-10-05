// Headless Vulkan compute spike for the QuestLHSync GPU solver port (Inc 2, phase 2).
// No surface / no swapchain: create an instance, pick a device with a compute queue,
// run a trivial vector-add SPIR-V kernel, and verify the result. If this runs, a
// compute-only Vulkan context is viable inside the SteamVR driver process.
//
// Build (clang++):  clang++ -std=c++17 -O2 vk_spike.cpp "<SDK>/Lib/vulkan-1.lib" -o vk_spike.exe
// Run:              vk_spike.exe vadd.spv

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  std::fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, (int)_r, __LINE__); return 2; } } while (0)

static std::vector<uint32_t> readSpv(const char *path) {
  FILE *f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(2); }
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> v(n / 4);
  if (std::fread(v.data(), 1, n, f) != (size_t)n) { std::exit(2); }
  std::fclose(f);
  return v;
}

static uint32_t findMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

int main(int argc, char **argv) {
  const char *spv = argc > 1 ? argv[1] : "vadd.spv";
  const uint32_t N = 1u << 20;  // 1,048,576 elements

  // --- instance (headless: no surface extensions) ---
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1;
  app.pApplicationName = "qlhs-vk-spike";
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

  // --- pick a physical device with a compute queue ---
  uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
  if (!nd) { std::fprintf(stderr, "no Vulkan physical devices\n"); return 3; }
  std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst, &nd, devs.data());
  VkPhysicalDevice pd = VK_NULL_HANDLE; uint32_t qf = 0;
  for (auto d : devs) {
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, q.data());
    for (uint32_t i = 0; i < nq; i++)
      if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        std::printf("device: %s  (compute queue family %u)\n", p.deviceName, i);
        if (pd == VK_NULL_HANDLE) { pd = d; qf = i; }
      }
  }
  if (pd == VK_NULL_HANDLE) { std::fprintf(stderr, "no compute queue\n"); return 3; }

  // --- logical device + queue ---
  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VkDevice dev; VK_CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

  // --- three host-visible storage buffers (a, b, c) ---
  const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(float);
  VkBuffer buf[3]; VkDeviceMemory mem[3]; void *map[3];
  for (int i = 0; i < 3; i++) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf[i]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf[i], &mr);
    uint32_t mt = findMem(pd, mr.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { std::fprintf(stderr, "no host-visible mem\n"); return 4; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    VK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem[i]));
    VK_CHECK(vkBindBufferMemory(dev, buf[i], mem[i], 0));
    VK_CHECK(vkMapMemory(dev, mem[i], 0, bytes, 0, &map[i]));
  }
  float *a = (float *)map[0], *b = (float *)map[1], *c = (float *)map[2];
  for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = 2.0f * (float)i; c[i] = -1.0f; }

  // --- descriptor set layout: 3 storage buffers ---
  VkDescriptorSetLayoutBinding b3[3]{};
  for (int i = 0; i < 3; i++) { b3[i].binding = i; b3[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b3[i].descriptorCount = 1; b3[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3; dslci.pBindings = b3;
  VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

  // --- shader module + compute pipeline ---
  std::vector<uint32_t> code = readSpv(spv);
  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = code.size() * 4; smci.pCode = code.data();
  VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
  cpci.layout = pl;
  VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  // --- descriptor pool + set ---
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &ps;
  VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo dbi[3]; VkWriteDescriptorSet w[3]{};
  for (int i = 0; i < 3; i++) {
    dbi[i] = {buf[i], 0, bytes};
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds; w[i].dstBinding = i;
    w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
  }
  vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

  // --- command buffer: dispatch ---
  VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cpci2.queueFamilyIndex = qf;
  VkCommandPool cp; VK_CHECK(vkCreateCommandPool(dev, &cpci2, nullptr, &cp));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
  vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
  VK_CHECK(vkEndCommandBuffer(cb));

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence; VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
  VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
  VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  // --- verify ---
  uint32_t bad = 0;
  for (uint32_t i = 0; i < N; i++) { float want = a[i] + b[i]; if (c[i] != want) { if (bad < 4)
    std::fprintf(stderr, "mismatch i=%u got %f want %f\n", i, c[i], want); bad++; } }
  std::printf("elements: %u   mismatches: %u\n", N, bad);
  std::printf(bad ? "SPIKE FAIL\n" : "SPIKE OK: headless Vulkan compute works\n");
  return bad ? 5 : 0;
}
