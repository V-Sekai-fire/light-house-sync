// Smoke test: drive the ported L-BFGS-B (lbfgsb.cpp) + CPU backend (vec_cpu, over the
// slang->C++ emits) on a bounded quadratic, and check it reaches the known optimum.
//   f(x) = sum (x_i - c_i)^2   over box [lo,hi];  argmin = clamp(c, lo, hi)
// Build: zig c++ -std=c++17 -O2 lbfgsb_test.cpp lbfgsb.cpp vec_cpu.cpp -o lbfgsb_test.exe
#include "lbfgsb.h"
#include "vec_cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

int main() {
  const uint32_t n = 4;
  const float c[4]  = { 1.0f, -2.0f, 0.5f, 3.0f };
  const float lo[4] = { -1, -1, -1, -1 };
  const float hi[4] = {  2,  2,  2,  2 };
  const float want[4] = { 1.0f, -1.0f, 0.5f, 2.0f };  // clamp(c, lo, hi)
  float x0[4] = { 0, 0, 0, 0 };

  VecCpu vec;
  Lbfgsb solver(vec);
  LbfgsbParams p;  // defaults
  std::string serr;

  Lbfgsb::Status st = solver.start(n, x0, lo, hi, p);
  std::vector<float> x;
  int guard = 0;
  while (st != Lbfgsb::CONVERGED && st != Lbfgsb::FAIL && guard++ < 100000) {
    if (st == Lbfgsb::NEED_EVAL || st == Lbfgsb::TRY) {
      if (!solver.readX(x)) { std::fprintf(stderr, "readX failed: %s\n", vec.error().c_str()); return 3; }
      double f = 0; std::vector<float> g(n);
      for (uint32_t i = 0; i < n; i++) { double d = (double)x[i] - c[i]; f += d * d; g[i] = (float)(2.0 * d); }
      solver.setGradient(g.data());
      st = solver.next(f);
    } else {
      st = solver.next();  // BUSY / ACCEPT
    }
  }

  std::printf("status=%s iters=%d nfev=%d fx=%.6g pg=%.3g reason=%s\n",
              Lbfgsb::status_name(st), solver.iterations(), solver.nfev(), solver.fx(),
              solver.pgNorm(), solver.reason().c_str());
  if (st == Lbfgsb::FAIL) { std::fprintf(stderr, "FAIL: %s\n", solver.error().c_str()); return 4; }

  solver.readX(x);
  double maxerr = 0;
  for (uint32_t i = 0; i < n; i++) { double e = std::fabs((double)x[i] - want[i]); if (e > maxerr) maxerr = e; }
  std::printf("x = [%.5f %.5f %.5f %.5f]  want [%.3f %.3f %.3f %.3f]  maxerr=%.3g\n",
              x[0], x[1], x[2], x[3], want[0], want[1], want[2], want[3], maxerr);
  bool ok = (st == Lbfgsb::CONVERGED) && (maxerr < 1e-3);
  std::printf(ok ? "LBFGSB TEST OK: ported solver reaches the bounded optimum\n"
                 : "LBFGSB TEST FAIL\n");
  return ok ? 0 : 5;
}
