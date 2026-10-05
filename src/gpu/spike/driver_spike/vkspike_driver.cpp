// SteamVR driver spike: proves a headless Vulkan compute context can be created and dispatched
// from INSIDE the vrserver.exe process (Inc 2, phase 2b). On Init() it runs an embedded vector-add
// SPIR-V kernel and writes the outcome to a log file. It registers no devices.
//
// Build (clang++):
//   clang++ -std=c++17 -O2 -shared -I<openvr>/headers -I<SDK>/Include \
//     vkspike_driver.cpp -L<SDK>/Lib -lvulkan-1 -o driver_vkspike.dll

#include <openvr_driver.h>
#include <vulkan/vulkan.h>
#include "vadd_spv.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <ctime>

static void logLine(const std::string &s) {
  std::string path;
  if (const char *la = std::getenv("LOCALAPPDATA")) path = std::string(la) + "\\QuestLHSync\\vkspike_driver.log";
  else path = "C:\\vkspike_driver.log";
  FILE *f = std::fopen(path.c_str(), "a");
  if (!f) { f = std::fopen("C:\\vkspike_driver.log", "a"); }
  if (!f) return;
  std::time_t t = std::time(nullptr);
  char ts[32]; std::strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  std::fprintf(f, "%s  %s\n", ts, s.c_str());
  std::fclose(f);
}

#define SPK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
  logLine(std::string("FAIL ") + #x + " = " + std::to_string((int)_r)); return false; } } while (0)

static uint32_t findMem(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

static bool vkSpike() {
  const uint32_t N = 1u << 20;
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1; app.pApplicationName = "qlhs-vkspike-driver";
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
  VkInstance inst; SPK(vkCreateInstance(&ici, nullptr, &inst));

  uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
  if (!nd) { logLine("no physical devices"); return false; }
  std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst, &nd, devs.data());
  VkPhysicalDevice pd = VK_NULL_HANDLE; uint32_t qf = 0;
  for (auto d : devs) {
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(d, &nq, q.data());
    for (uint32_t i = 0; i < nq; i++) if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d, &p);
      logLine(std::string("device: ") + p.deviceName + " (compute qf " + std::to_string(i) + ")");
      if (pd == VK_NULL_HANDLE) { pd = d; qf = i; } break; }
  }
  if (pd == VK_NULL_HANDLE) { logLine("no compute queue"); return false; }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
  VkDevice dev; SPK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

  const VkDeviceSize bytes = (VkDeviceSize)N * sizeof(float);
  VkBuffer buf[3]; VkDeviceMemory mem[3]; void *map[3];
  for (int i = 0; i < 3; i++) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    SPK(vkCreateBuffer(dev, &bci, nullptr, &buf[i]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf[i], &mr);
    uint32_t mt = findMem(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) { logLine("no host-visible mem"); return false; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize = mr.size; mai.memoryTypeIndex = mt;
    SPK(vkAllocateMemory(dev, &mai, nullptr, &mem[i]));
    SPK(vkBindBufferMemory(dev, buf[i], mem[i], 0));
    SPK(vkMapMemory(dev, mem[i], 0, bytes, 0, &map[i]));
  }
  float *a = (float *)map[0], *b = (float *)map[1], *c = (float *)map[2];
  for (uint32_t i = 0; i < N; i++) { a[i] = (float)i; b[i] = 2.0f * (float)i; c[i] = -1.0f; }

  VkDescriptorSetLayoutBinding b3[3]{};
  for (int i = 0; i < 3; i++) { b3[i].binding = i; b3[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b3[i].descriptorCount = 1; b3[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dslci.bindingCount = 3; dslci.pBindings = b3;
  VkDescriptorSetLayout dsl; SPK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount = 1; plci.pSetLayouts = &dsl; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
  VkPipelineLayout pl; SPK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

  VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = (size_t)vadd_spv_words * 4; smci.pCode = vadd_spv;
  VkShaderModule sm; SPK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main"; cpci.layout = pl;
  VkPipeline pipe; SPK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
  VkDescriptorPool dp; SPK(vkCreateDescriptorPool(dev, &dpci, nullptr, &dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dsl;
  VkDescriptorSet dset; SPK(vkAllocateDescriptorSets(dev, &dsai, &dset));
  VkDescriptorBufferInfo dbi[3]; VkWriteDescriptorSet w[3]{};
  for (int i = 0; i < 3; i++) { dbi[i] = {buf[i], 0, bytes};
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = dset; w[i].dstBinding = i;
    w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i]; }
  vkUpdateDescriptorSets(dev, 3, w, 0, nullptr);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex = qf;
  VkCommandPool cp; SPK(vkCreateCommandPool(dev, &cpi, nullptr, &cp));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cbai.commandPool = cp; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
  VkCommandBuffer cb; SPK(vkAllocateCommandBuffers(dev, &cbai, &cb));
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  SPK(vkBeginCommandBuffer(cb, &cbbi));
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &dset, 0, nullptr);
  vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &N);
  vkCmdDispatch(cb, (N + 63) / 64, 1, 1);
  SPK(vkEndCommandBuffer(cb));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence; SPK(vkCreateFence(dev, &fci, nullptr, &fence));
  SPK(vkQueueSubmit(queue, 1, &si, fence));
  SPK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  uint32_t bad = 0;
  for (uint32_t i = 0; i < N; i++) if (c[i] != a[i] + b[i]) bad++;
  logLine("elements " + std::to_string(N) + "  mismatches " + std::to_string(bad));
  logLine(bad ? "SPIKE FAIL (compute mismatch)" : "SPIKE OK: headless Vulkan compute works INSIDE vrserver");
  return bad == 0;
}

namespace {
class Provider : public vr::IServerTrackedDeviceProvider {
public:
  vr::EVRInitError Init(vr::IVRDriverContext *) override {
    logLine("==== vkspike driver Init() in vrserver process ====");
    bool ok = vkSpike();
    logLine(ok ? "Init: vkSpike returned OK" : "Init: vkSpike returned FAIL");
    return vr::VRInitError_None;  // register nothing; we only wanted the spike to run
  }
  void Cleanup() override { logLine("Cleanup()"); }
  const char *const *GetInterfaceVersions() override { return vr::k_InterfaceVersions; }
  void RunFrame() override {}
  bool ShouldBlockStandbyMode() override { return false; }
  void EnterStandby() override {}
  void LeaveStandby() override {}
};
Provider g_provider;
}

extern "C" __declspec(dllexport)
void *HmdDriverFactory(const char *interfaceName, vr::EVRInitError *error) {
  if (std::strcmp(interfaceName, vr::IServerTrackedDeviceProvider_Version) == 0) return &g_provider;
  if (error) *error = vr::VRInitError_Init_InterfaceNotFound;
  return nullptr;
}
