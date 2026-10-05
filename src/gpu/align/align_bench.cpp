// Benchmark: one full alignment f+gradient evaluation (9 cost variants = central-diff grad),
// CPU (align_objective) vs GPU (align_residual df32 kernel), at a realistic sighting count.
// This is the per-eval cost the L-BFGS-B driver pays each line-search trial.
// Build: zig c++ -std=c++17 -O2 -I<SDK>/Include align_bench.cpp ../gpu_compute.cpp -L<SDK>/Lib -lvulkan-1 -o align_bench.exe
#include "align_objective.h"
#include "../gpu_compute.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>

using namespace qlhs::align;
using qlhs::gpu::GpuCompute; using qlhs::gpu::Buffer; using qlhs::gpu::Kernel;
using clk = std::chrono::high_resolution_clock;

static std::vector<uint32_t> readSpv(const char*p){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"open %s\n",p);exit(2);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<uint32_t>v(n/4);if(fread(v.data(),1,n,f)!=(size_t)n)exit(2);fclose(f);return v;}
struct GP{float y[4];float h;uint32_t n,p0,p1;};

int main(int argc,char**argv){
  const char* spv = argc>1?argv[1]:"../kernels_spv/align_residual.spv";
  const uint32_t N = argc>2?(uint32_t)atoi(argv[2]):100000;   // sightings
  const int REPS = 200;
  const double H=1e-4;

  // synth: 3 stations, N sightings with jitter around true rays
  Problem prob; prob.S={{0,2,0},{3,2,-1},{-2,2,2}};
  const double yawS=0.6; const V3 tS{0.5,-0.3,1.2}; double c=cos(yawS),s=sin(yawS);
  srand(1);
  for(uint32_t i=0;i<N;i++){ int k=i%3; V3 o{ (rand()/(double)RAND_MAX-0.5)*1.0, 1.5+(rand()/(double)RAND_MAX)*0.3, (rand()/(double)RAND_MAX-0.5)*1.0 };
    V3 Sq=ryMul(c,s,prob.S[k])+tS; V3 d=Sq-o; double nd=norm(d); d={d.x/nd,d.y/nd,d.z/nd}; prob.rays.push_back({k,o,d,1.0}); }

  double y[4]={0.3,0.2,-0.1,0.5};

  // --- CPU: f + central-diff grad (9 cost evals) ---
  volatile double sink=0;
  auto t0=clk::now();
  for(int r=0;r<REPS;r++){ double g[4]; double f=prob.cost(y); prob.grad(y,g); sink+=f+g[0]+g[1]+g[2]+g[3]; }
  auto t1=clk::now();
  double cpu_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/REPS;

  // --- GPU setup ---
  std::string err; GpuCompute gc; if(!gc.init(err)){fprintf(stderr,"init %s\n",err.c_str());return 2;}
  auto code=readSpv(spv);
  Kernel k=gc.loadKernel(code.data(),code.size(),{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},1,&err);
  Buffer params=gc.alloc(sizeof(GP),true,&err), Sb=gc.alloc(prob.S.size()*16,false,&err), rk=gc.alloc((VkDeviceSize)N*4,false,&err),
         ro=gc.alloc((VkDeviceSize)N*16,false,&err), rd=gc.alloc((VkDeviceSize)N*16,false,&err), rsw=gc.alloc((VkDeviceSize)N*4,false,&err), dst=gc.alloc(18*4,false,&err);
  {float*S4=(float*)Sb.map;for(size_t i=0;i<prob.S.size();i++){S4[4*i]=(float)prob.S[i].x;S4[4*i+1]=(float)prob.S[i].y;S4[4*i+2]=(float)prob.S[i].z;S4[4*i+3]=0;}}
  {uint32_t*K=(uint32_t*)rk.map;float*O=(float*)ro.map;float*D=(float*)rd.map;float*W=(float*)rsw.map;for(uint32_t i=0;i<N;i++){auto&s2=prob.rays[i];K[i]=s2.k;O[4*i]=(float)s2.o.x;O[4*i+1]=(float)s2.o.y;O[4*i+2]=(float)s2.o.z;O[4*i+3]=0;D[4*i]=(float)s2.d.x;D[4*i+1]=(float)s2.d.y;D[4*i+2]=(float)s2.d.z;D[4*i+3]=0;W[i]=1;}}
  auto gpuEval=[&](){ GP*pp=(GP*)params.map; for(int i=0;i<4;i++)pp->y[i]=(float)y[i]; pp->h=(float)H; pp->n=N; pp->p0=0;pp->p1=0;
    std::string e; gc.run({{&k,{&params,&Sb,&rk,&ro,&rd,&rsw,&dst},1}},&e); float*o=(float*)dst.map; return (double)o[0]+o[1]; };
  gpuEval(); // warmup
  auto t2=clk::now();
  for(int r=0;r<REPS;r++) sink+=gpuEval();
  auto t3=clk::now();
  double gpu_ms = std::chrono::duration<double,std::milli>(t3-t2).count()/REPS;

  printf("device: %s   sightings=%u   reps=%d\n", gc.deviceName(), N, REPS);
  printf("CPU f+grad: %.3f ms/eval\n", cpu_ms);
  printf("GPU f+grad: %.3f ms/eval  (1 dispatch, 9 variants, df32 reduce + readback)\n", gpu_ms);
  printf("speedup: %.1fx\n", cpu_ms/gpu_ms);
  (void)sink;
  return 0;
}
