// Real-kernel GPU dispatch spike: run lbfgsb's saxpby.slang (emitted to SPIR-V) through a
// headless Vulkan context, proving the actual kernel ABI (UBO params at binding 0 + SSBOs at
// 1..3, numthreads 256). This is the dispatch convention the full L-BFGS-B vec backend uses.
//   dst[i] = alpha*x[i] + beta*y[i]
//
// Build:  zig cc -std=c++17 -O2 -I<SDK>/Include saxpby_gpu.cpp -L<SDK>/Lib -lvulkan-1 -o saxpby_gpu.exe
// Run:    saxpby_gpu.exe kernels_spv/saxpby.spv

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define VK_CHECK(x) do { VkResult _r=(x); if(_r!=VK_SUCCESS){ std::fprintf(stderr,"FAIL %s=%d (L%d)\n",#x,(int)_r,__LINE__); return 2;} } while(0)

struct Params { uint32_t n; float alpha; float beta; };

static std::vector<uint32_t> readSpv(const char *p){ FILE*f=std::fopen(p,"rb"); if(!f){std::fprintf(stderr,"open %s\n",p);std::exit(2);} std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET); std::vector<uint32_t> v(n/4); if(std::fread(v.data(),1,n,f)!=(size_t)n)std::exit(2); std::fclose(f); return v; }
static uint32_t findMem(VkPhysicalDevice pd,uint32_t bits,VkMemoryPropertyFlags want){ VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp); for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((bits&(1u<<i))&&(mp.memoryTypes[i].propertyFlags&want)==want) return i; return UINT32_MAX; }

int main(int argc,char**argv){
  const char* spv = argc>1?argv[1]:"saxpby.spv";
  const uint32_t N = 1u<<20; const float ALPHA=3.0f, BETA=-2.0f;

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion=VK_API_VERSION_1_1;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&app;
  VkInstance inst; VK_CHECK(vkCreateInstance(&ici,nullptr,&inst));
  uint32_t nd=0; vkEnumeratePhysicalDevices(inst,&nd,nullptr); if(!nd){std::fprintf(stderr,"no device\n");return 3;}
  std::vector<VkPhysicalDevice> devs(nd); vkEnumeratePhysicalDevices(inst,&nd,devs.data());
  VkPhysicalDevice pd=VK_NULL_HANDLE; uint32_t qf=0;
  for(auto d:devs){ uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(d,&nq,nullptr); std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(d,&nq,q.data());
    for(uint32_t i=0;i<nq;i++) if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT){ if(pd==VK_NULL_HANDLE){pd=d;qf=i;} break; } }
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd,&pp); std::printf("device: %s\n",pp.deviceName);
  float prio=1.0f; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueFamilyIndex=qf; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci;
  VkDevice dev; VK_CHECK(vkCreateDevice(pd,&dci,nullptr,&dev)); VkQueue queue; vkGetDeviceQueue(dev,qf,0,&queue);

  // buffers: [0]=params UBO, [1]=x, [2]=y, [3]=dst (SSBO)
  VkDeviceSize vb=(VkDeviceSize)N*sizeof(float);
  VkDeviceSize sizes[4]={sizeof(Params),vb,vb,vb};
  VkBufferUsageFlags usage[4]={VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
  VkBuffer buf[4]; VkDeviceMemory mem[4]; void* map[4];
  for(int i=0;i<4;i++){ VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=sizes[i]; bci.usage=usage[i]; bci.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev,&bci,nullptr,&buf[i])); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf[i],&mr);
    uint32_t mt=findMem(pd,mr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); if(mt==UINT32_MAX){std::fprintf(stderr,"no mem\n");return 4;}
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=mt;
    VK_CHECK(vkAllocateMemory(dev,&mai,nullptr,&mem[i])); VK_CHECK(vkBindBufferMemory(dev,buf[i],mem[i],0)); VK_CHECK(vkMapMemory(dev,mem[i],0,sizes[i],0,&map[i])); }
  *(Params*)map[0]=Params{N,ALPHA,BETA};
  float* x=(float*)map[1]; float* y=(float*)map[2]; float* dst=(float*)map[3];
  for(uint32_t i=0;i<N;i++){ x[i]=(float)(i%97); y[i]=(float)(i%13); dst[i]=-1.0f; }

  VkDescriptorSetLayoutBinding bnd[4]{};
  VkDescriptorType types[4]={VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  for(int i=0;i<4;i++){ bnd[i].binding=i; bnd[i].descriptorType=types[i]; bnd[i].descriptorCount=1; bnd[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT; }
  VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dslci.bindingCount=4; dslci.pBindings=bnd;
  VkDescriptorSetLayout dsl; VK_CHECK(vkCreateDescriptorSetLayout(dev,&dslci,nullptr,&dsl));
  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; plci.setLayoutCount=1; plci.pSetLayouts=&dsl;
  VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(dev,&plci,nullptr,&pl));
  std::vector<uint32_t> code=readSpv(spv); VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smci.codeSize=code.size()*4; smci.pCode=code.data();
  VkShaderModule sm; VK_CHECK(vkCreateShaderModule(dev,&smci,nullptr,&sm));
  VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module=sm; cpci.stage.pName="main"; cpci.layout=pl;
  VkPipeline pipe; VK_CHECK(vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpci,nullptr,&pipe));

  VkDescriptorPoolSize dps[2]={{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,1},{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};
  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpci.maxSets=1; dpci.poolSizeCount=2; dpci.pPoolSizes=dps;
  VkDescriptorPool dp; VK_CHECK(vkCreateDescriptorPool(dev,&dpci,nullptr,&dp));
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsai.descriptorPool=dp; dsai.descriptorSetCount=1; dsai.pSetLayouts=&dsl;
  VkDescriptorSet ds; VK_CHECK(vkAllocateDescriptorSets(dev,&dsai,&ds));
  VkDescriptorBufferInfo dbi[4]; VkWriteDescriptorSet w[4]{};
  for(int i=0;i<4;i++){ dbi[i]={buf[i],0,sizes[i]}; w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet=ds; w[i].dstBinding=i; w[i].descriptorCount=1; w[i].descriptorType=types[i]; w[i].pBufferInfo=&dbi[i]; }
  vkUpdateDescriptorSets(dev,4,w,0,nullptr);

  VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpi.queueFamilyIndex=qf;
  VkCommandPool cp; VK_CHECK(vkCreateCommandPool(dev,&cpi,nullptr,&cp));
  VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cbai.commandPool=cp; cbai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount=1;
  VkCommandBuffer cb; VK_CHECK(vkAllocateCommandBuffers(dev,&cbai,&cb));
  VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; cbbi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK(vkBeginCommandBuffer(cb,&cbbi));
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe);
  vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,&ds,0,nullptr);
  vkCmdDispatch(cb,(N+255)/256,1,1);
  VK_CHECK(vkEndCommandBuffer(cb));
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cb;
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; VK_CHECK(vkCreateFence(dev,&fci,nullptr,&fence));
  VK_CHECK(vkQueueSubmit(queue,1,&si,fence)); VK_CHECK(vkWaitForFences(dev,1,&fence,VK_TRUE,UINT64_MAX));

  uint32_t bad=0; for(uint32_t i=0;i<N;i++){ float want=std::fma(ALPHA,x[i],BETA*y[i]); if(dst[i]!=want){ if(bad<4) std::fprintf(stderr,"i=%u got %f want %f\n",i,dst[i],want); bad++; } }
  std::printf("saxpby N=%u alpha=%g beta=%g  mismatches=%u\n",N,ALPHA,BETA,bad);
  std::printf(bad?"KERNEL SPIKE FAIL\n":"KERNEL SPIKE OK: real lbfgsb saxpby runs on GPU\n");
  return bad?5:0;
}
