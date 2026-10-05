// GPU residual validation: (1) parity of the Slang df32 residual kernel vs the CPU objective,
// (2) full L-BFGS-B solve driven by the GPU residual recovering a known alignment.
// Build: zig c++ -std=c++17 -O2 -I<SDK>/Include align_gpu_test.cpp ../gpu_compute.cpp \
//        ../lbfgsb/lbfgsb.cpp ../lbfgsb/vec_cpu.cpp -L<SDK>/Lib -lvulkan-1 -o align_gpu_test.exe
#include "align_objective.h"
#include "../gpu_compute.h"
#include "../lbfgsb/lbfgsb.h"
#include "../lbfgsb/vec_cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace qlhs::align;
using qlhs::gpu::GpuCompute; using qlhs::gpu::Buffer; using qlhs::gpu::Kernel;

static std::vector<uint32_t> readSpv(const char *p){ FILE*f=std::fopen(p,"rb"); if(!f){std::fprintf(stderr,"open %s\n",p);std::exit(2);} std::fseek(f,0,SEEK_END); long n=std::ftell(f); std::fseek(f,0,SEEK_SET); std::vector<uint32_t> v(n/4); if(std::fread(v.data(),1,n,f)!=(size_t)n)std::exit(2); std::fclose(f); return v; }

struct GP { float y[4]; float h; uint32_t n, pad0, pad1; };

int main(int argc, char** argv) {
  const char* spv = argc>1?argv[1]:"../kernels_spv/align_residual.spv";
  const double H = 1e-4;

  // --- synthetic problem (same as the CPU test) ---
  const double yawStar=0.6; const V3 tStar{0.5,-0.3,1.2};
  Problem prob; prob.S = { {0,2,0},{3,2,-1},{-2,2,2} };
  std::vector<V3> cams = {{0,1.6,0},{0.5,1.6,0.2},{-0.4,1.5,0.1},{0.2,1.7,-0.3},{-0.3,1.6,0.4},{0.1,1.55,0.5},{0.4,1.65,-0.2},{-0.2,1.6,-0.1}};
  double c=std::cos(yawStar), s=std::sin(yawStar);
  for(size_t k=0;k<prob.S.size();k++) for(const V3&o:cams){ V3 Sq=ryMul(c,s,prob.S[k])+tStar; V3 d=Sq-o; double nd=norm(d); d={d.x/nd,d.y/nd,d.z/nd}; prob.rays.push_back({(int)k,o,d,1.0}); }
  const uint32_t nStat=(uint32_t)prob.S.size(), nSight=(uint32_t)prob.rays.size();

  // --- GPU setup ---
  std::string err; GpuCompute gc; if(!gc.init(err)){ std::fprintf(stderr,"init: %s\n",err.c_str()); return 2; }
  std::printf("device: %s  sightings=%u\n", gc.deviceName(), nSight);
  auto code=readSpv(spv);
  std::vector<VkDescriptorType> binds = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
  Kernel kern = gc.loadKernel(code.data(), code.size(), binds, /*localX=*/1, &err);
  if(!kern.pipe){ std::fprintf(stderr,"loadKernel: %s\n",err.c_str()); return 3; }

  Buffer params = gc.alloc(sizeof(GP), true, &err);
  Buffer Sb  = gc.alloc((VkDeviceSize)nStat*16, false, &err);
  Buffer rk  = gc.alloc((VkDeviceSize)nSight*4, false, &err);
  Buffer ro  = gc.alloc((VkDeviceSize)nSight*16, false, &err);
  Buffer rd  = gc.alloc((VkDeviceSize)nSight*16, false, &err);
  Buffer rsw = gc.alloc((VkDeviceSize)nSight*4, false, &err);
  Buffer dst = gc.alloc(18*4, false, &err);
  if(!dst.buf){ std::fprintf(stderr,"alloc: %s\n",err.c_str()); return 4; }
  { float* S4=(float*)Sb.map; for(uint32_t k=0;k<nStat;k++){ S4[4*k]=(float)prob.S[k].x; S4[4*k+1]=(float)prob.S[k].y; S4[4*k+2]=(float)prob.S[k].z; S4[4*k+3]=0; } }
  { uint32_t* K=(uint32_t*)rk.map; float* O=(float*)ro.map; float* D=(float*)rd.map; float* W=(float*)rsw.map;
    for(uint32_t i=0;i<nSight;i++){ const Sighting&s2=prob.rays[i]; K[i]=(uint32_t)s2.k; O[4*i]=(float)s2.o.x;O[4*i+1]=(float)s2.o.y;O[4*i+2]=(float)s2.o.z;O[4*i+3]=0; D[4*i]=(float)s2.d.x;D[4*i+1]=(float)s2.d.y;D[4*i+2]=(float)s2.d.z;D[4*i+3]=0; W[i]=(float)s2.sw; } }

  auto gpuEval=[&](const double* y, double* g)->double{
    GP* pp=(GP*)params.map; for(int i=0;i<4;i++) pp->y[i]=(float)y[i]; pp->h=(float)H; pp->n=nSight; pp->pad0=0; pp->pad1=0;
    std::string e; gc.run({{&kern,{&params,&Sb,&rk,&ro,&rd,&rsw,&dst},1}}, &e);
    float* o=(float*)dst.map; auto df=[&](int pr){ return (double)o[2*pr]+(double)o[2*pr+1]; };
    if(g) for(int j=0;j<4;j++) g[j]=(df(1+2*j)-df(2+2*j))/(2*H);
    return df(0);
  };

  // --- (1) parity GPU vs CPU at a few points ---
  double maxfe=0, maxge=0;
  double testY[3][4] = {{0,0,0,0},{0.3,0.2,-0.1,0.5},{yawStar,tStar.x,tStar.y,tStar.z}};
  for(auto& y : testY){
    double gg[4]; double fg=gpuEval(y,gg);
    double fc=prob.cost(y); double gc4[4]; prob.grad(y,gc4);
    double fe=std::fabs(fg-fc)/(1+std::fabs(fc)); maxfe=std::max(maxfe,fe);
    for(int j=0;j<4;j++){ double ge=std::fabs(gg[j]-gc4[j])/(1+std::fabs(gc4[j])); maxge=std::max(maxge,ge); }
    std::printf("y=[%.2f %.2f %.2f %.2f]  f_gpu=%.6e f_cpu=%.6e  relfe=%.2e relge=%.2e\n", y[0],y[1],y[2],y[3],fg,fc,fe,maxge);
  }
  std::printf("parity: max rel f err=%.2e  max rel g err=%.2e\n", maxfe, maxge);

  // --- (2) full solve driven by the GPU residual ---
  VecCpu vec; Lbfgsb solver(vec); LbfgsbParams p; p.max_iterations=200;
  const float lo[4]={-3.14159f,-10,-10,-10}, hi[4]={3.14159f,10,10,10}; float x0[4]={0,0,0,0};
  Lbfgsb::Status st=solver.start(4,x0,lo,hi,p); std::vector<float> xf; int guard=0;
  while(st!=Lbfgsb::CONVERGED && st!=Lbfgsb::FAIL && guard++<200000){
    if(st==Lbfgsb::NEED_EVAL||st==Lbfgsb::TRY){ solver.readX(xf); double y[4]={xf[0],xf[1],xf[2],xf[3]}; double g[4]; double f=gpuEval(y,g); float gf[4]={(float)g[0],(float)g[1],(float)g[2],(float)g[3]}; solver.setGradient(gf); st=solver.next(f); }
    else st=solver.next();
  }
  solver.readX(xf); double yaw=wrap(xf[0]);
  double eyaw=std::fabs(wrap(yaw-yawStar));
  double et=std::sqrt((xf[1]-tStar.x)*(xf[1]-tStar.x)+(xf[2]-tStar.y)*(xf[2]-tStar.y)+(xf[3]-tStar.z)*(xf[3]-tStar.z));
  std::printf("solve: status=%s iters=%d fx=%.3e  yaw=%.5f(%.5f) t=[%.4f %.4f %.4f]  yawerr=%.2e terr=%.2e\n",
              Lbfgsb::status_name(st), solver.iterations(), solver.fx(), yaw, yawStar, xf[1],xf[2],xf[3], eyaw, et);
  bool ok = (maxfe<1e-4) && (st==Lbfgsb::CONVERGED) && eyaw<2e-3 && et<2e-3;
  std::printf(ok ? "ALIGN GPU OK: df32 residual matches CPU and the GPU-driven solve recovers the alignment\n"
                 : "ALIGN GPU FAIL\n");
  return ok?0:5;
}
