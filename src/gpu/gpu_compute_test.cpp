// Smoke test for GpuCompute: run lbfgsb saxpby through the reusable device layer.
//   dst[i] = alpha*x[i] + beta*y[i]
// Build: zig c++ -std=c++17 -O2 -I<SDK>/Include gpu_compute_test.cpp gpu_compute.cpp -L<SDK>/Lib -lvulkan-1 -o gpu_compute_test.exe
// Run:   gpu_compute_test.exe kernels_spv/saxpby.spv
#include "gpu_compute.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace qlhs::gpu;

static std::vector<uint32_t> readSpv(const char *p) {
  FILE *f = std::fopen(p, "rb"); if (!f) { std::fprintf(stderr, "open %s\n", p); std::exit(2); }
  std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
  std::vector<uint32_t> v(n / 4); if (std::fread(v.data(), 1, n, f) != (size_t)n) std::exit(2);
  std::fclose(f); return v;
}

struct Params { uint32_t n; float alpha; float beta; };

int main(int argc, char **argv) {
  const char *spv = argc > 1 ? argv[1] : "kernels_spv/saxpby.spv";
  const uint32_t N = 1u << 20; const float A = 3.0f, B = -2.0f;
  std::string err;
  GpuCompute gc;
  if (!gc.init(err)) { std::fprintf(stderr, "init: %s\n", err.c_str()); return 2; }
  std::printf("device: %s\n", gc.deviceName());

  auto code = readSpv(spv);
  Kernel k = gc.loadKernel(code.data(), code.size(),
    {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 256, &err);
  if (!k.pipe) { std::fprintf(stderr, "loadKernel: %s\n", err.c_str()); return 3; }

  Buffer params = gc.alloc(sizeof(Params), /*uniform=*/true, &err);
  Buffer bx = gc.alloc((VkDeviceSize)N * sizeof(float), false, &err);
  Buffer by = gc.alloc((VkDeviceSize)N * sizeof(float), false, &err);
  Buffer bd = gc.alloc((VkDeviceSize)N * sizeof(float), false, &err);
  if (!bd.buf) { std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 4; }
  *(Params *)params.map = Params{N, A, B};
  float *x = (float *)bx.map, *y = (float *)by.map, *d = (float *)bd.map;
  for (uint32_t i = 0; i < N; i++) { x[i] = (float)(i % 97); y[i] = (float)(i % 13); d[i] = -1.0f; }

  if (!gc.run({{&k, {&params, &bx, &by, &bd}, N}}, &err)) { std::fprintf(stderr, "run: %s\n", err.c_str()); return 5; }

  uint32_t bad = 0;
  for (uint32_t i = 0; i < N; i++) { float want = std::fma(A, x[i], B * y[i]); if (d[i] != want) bad++; }
  std::printf("saxpby via GpuCompute  N=%u  mismatches=%u\n", N, bad);
  std::printf(bad ? "GPUCOMPUTE TEST FAIL\n" : "GPUCOMPUTE TEST OK: device layer drives real kernels\n");
  gc.free(params); gc.free(bx); gc.free(by); gc.free(bd); gc.destroyKernel(k); gc.shutdown();
  return bad ? 6 : 0;
}
