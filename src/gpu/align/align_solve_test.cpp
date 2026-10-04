// Standalone validation: synthesize base-station sightings for a known alignment, then recover
// it with the ported L-BFGS-B over the alignment objective. No sync.cpp, no GPU yet — proves the
// objective + solver recover (yaw, t) before wiring into the driver.
// Build: zig c++ -std=c++17 -O2 align_solve_test.cpp ../lbfgsb/lbfgsb.cpp ../lbfgsb/vec_cpu.cpp -o align_solve_test.exe
#include "align_objective.h"
#include "../lbfgsb/lbfgsb.h"
#include "../lbfgsb/vec_cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace qlhs::align;

int main() {
  // --- ground truth ---
  const double yawStar = 0.6;                 // rad
  const V3 tStar{0.5, -0.3, 1.2};             // m
  Problem prob;
  prob.S = { {0, 2.0, 0}, {3.0, 2.0, -1.0}, {-2.0, 2.0, 2.0} };   // 3 stations, reference frame

  // cameras (quest-space origins) that "see" the stations
  std::vector<V3> cams = {
    {0, 1.6, 0}, {0.5, 1.6, 0.2}, {-0.4, 1.5, 0.1}, {0.2, 1.7, -0.3},
    {-0.3, 1.6, 0.4}, {0.1, 1.55, 0.5}, {0.4, 1.65, -0.2}, {-0.2, 1.6, -0.1},
  };
  const double c = std::cos(yawStar), s = std::sin(yawStar);
  for (size_t k = 0; k < prob.S.size(); k++)
    for (const V3 &o : cams) {
      V3 Sq = ryMul(c, s, prob.S[k]) + tStar;     // station in quest space at truth
      V3 d = Sq - o; double nd = norm(d); d = {d.x / nd, d.y / nd, d.z / nd};
      prob.rays.push_back({(int)k, o, d, 1.0});
    }
  std::printf("stations=%zu cameras=%zu sightings=%zu\n", prob.S.size(), cams.size(), prob.rays.size());

  // --- solve with ported L-BFGS-B ---
  VecCpu vec; Lbfgsb solver(vec); LbfgsbParams p; p.max_iterations = 200;
  const float lo[4] = { -3.14159f, -10, -10, -10 };
  const float hi[4] = {  3.14159f,  10,  10,  10 };
  float x0[4] = { 0, 0, 0, 0 };               // cold start

  Lbfgsb::Status st = solver.start(4, x0, lo, hi, p);
  std::vector<float> xf; int guard = 0;
  while (st != Lbfgsb::CONVERGED && st != Lbfgsb::FAIL && guard++ < 200000) {
    if (st == Lbfgsb::NEED_EVAL || st == Lbfgsb::TRY) {
      solver.readX(xf);
      double y[4] = { xf[0], xf[1], xf[2], xf[3] };
      double f = prob.cost(y), g[4]; prob.grad(y, g);
      float gf[4] = { (float)g[0], (float)g[1], (float)g[2], (float)g[3] };
      solver.setGradient(gf);
      st = solver.next(f);
    } else st = solver.next();
  }
  solver.readX(xf);
  double yaw = wrap(xf[0]);
  std::printf("status=%s iters=%d nfev=%d fx=%.3e reason=%s\n",
              Lbfgsb::status_name(st), solver.iterations(), solver.nfev(), solver.fx(), solver.reason().c_str());
  std::printf("recovered yaw=%.5f (truth %.5f)  t=[%.5f %.5f %.5f] (truth [%.3f %.3f %.3f])\n",
              yaw, yawStar, xf[1], xf[2], xf[3], tStar.x, tStar.y, tStar.z);

  double eyaw = std::fabs(wrap(yaw - yawStar));
  double et = std::sqrt((xf[1]-tStar.x)*(xf[1]-tStar.x) + (xf[2]-tStar.y)*(xf[2]-tStar.y) + (xf[3]-tStar.z)*(xf[3]-tStar.z));
  std::printf("yaw err=%.2e rad  |t err|=%.2e m\n", eyaw, et);
  bool ok = (st == Lbfgsb::CONVERGED) && eyaw < 1e-3 && et < 1e-3;
  std::printf(ok ? "ALIGN SOLVE OK: recovered the known alignment\n" : "ALIGN SOLVE FAIL\n");
  return ok ? 0 : 5;
}
