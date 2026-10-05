// Validate figure-8 paired calibration: synthesize a held tracker+controller waving a figure-8 for a
// known transform (yaw,t) + mount offset m, then (1) solve from the whole capture, and (2) run it
// CONTINUOUSLY on a rolling window as samples stream in, showing it converges and stays locked.
// Build: zig c++ -std=c++17 -O2 align_figure8_test.cpp ../lbfgsb/lbfgsb.cpp ../lbfgsb/vec_cpu.cpp -o align_figure8_test.exe
#include "align_figure8.h"
#include "../lbfgsb/lbfgsb.h"
#include "../lbfgsb/vec_cpu.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace qlhs::align;

static double frand(double a, double b) { return a + (b - a) * (rand() / (double)RAND_MAX); }

// solve y[7] from the samples in prob; returns final cost
static double solve8(const Figure8 &prob, double *yout) {
  VecCpu vec; Lbfgsb solver(vec); LbfgsbParams p; p.max_iterations = 300;
  const float lo[7] = { -6.3f, -10,-10,-10, -0.5f,-0.5f,-0.5f };
  const float hi[7] = {  6.3f,  10, 10, 10,  0.5f, 0.5f, 0.5f };
  float x0[7] = {0,0,0,0,0,0,0};
  auto st = solver.start(7, x0, lo, hi, p);
  std::vector<float> xf; int guard = 0;
  while (st != Lbfgsb::CONVERGED && st != Lbfgsb::FAIL && guard++ < 200000) {
    if (st == Lbfgsb::NEED_EVAL || st == Lbfgsb::TRY) {
      solver.readX(xf);
      double y[7]; for (int j=0;j<7;j++) y[j]=xf[j];
      double f = prob.cost(y), g[7]; prob.grad(y, g);
      float gf[7]; for (int j=0;j<7;j++) gf[j]=(float)g[j];
      solver.setGradient(gf); st = solver.next(f);
    } else st = solver.next();
  }
  solver.readX(xf); for (int j=0;j<7;j++) yout[j]=xf[j];
  double y[7]; for (int j=0;j<7;j++) y[j]=yout[j]; return prob.cost(y);
}

int main() {
  srand(5);
  const double yawStar = 0.6; const V3 tStar{0.5,-0.3,1.2}; const V3 mStar{0.08,-0.03,0.05};
  const double cS = std::cos(yawStar), sS = std::sin(yawStar);

  // build a full figure-8 capture (tracker pose streams; controller derived from the rigid link)
  std::vector<Sample8> stream;
  const int N = 120;
  for (int i = 0; i < N; i++) {
    double u = 2*3.14159265 * i / N;
    V3 b{ 0.8*std::sin(u), 1.2 + 0.25*std::sin(2*u), 0.8*std::sin(u)*std::cos(u) };   // lemniscate
    M3 Rb = m3matmul(m3matmul(rotY(0.6*std::sin(u)), rotX(0.5*std::cos(u))), rotZ(0.4*std::sin(2*u)));
    V3 a = ryMul(cS, sS, b + m3mul(Rb, mStar)) + tStar;
    a.x += frand(-0.001,0.001); a.y += frand(-0.001,0.001); a.z += frand(-0.001,0.001);  // 1mm noise
    stream.push_back({a, b, Rb});
  }

  auto report = [&](const char *tag, const double *y) {
    double ey = std::fabs(wrap(y[0]-yawStar));
    double et = std::sqrt((y[1]-tStar.x)*(y[1]-tStar.x)+(y[2]-tStar.y)*(y[2]-tStar.y)+(y[3]-tStar.z)*(y[3]-tStar.z));
    double em = std::sqrt((y[4]-mStar.x)*(y[4]-mStar.x)+(y[5]-mStar.y)*(y[5]-mStar.y)+(y[6]-mStar.z)*(y[6]-mStar.z));
    std::printf("%s yaw=%.4f(%.4f) t=[%.3f %.3f %.3f] m=[%.3f %.3f %.3f]  yawerr=%.2e terr=%.2e merr=%.2e\n",
                tag, wrap(y[0]), yawStar, y[1],y[2],y[3], y[4],y[5],y[6], ey, et, em);
    return ey<5e-3 && et<5e-3 && em<5e-2;
  };

  // (1) one-shot over the whole capture
  Figure8 full; full.s = stream; double yf[7]; solve8(full, yf);
  bool ok1 = report("one-shot:  ", yf);

  // (2) CONTINUOUS: rolling window of the most recent W samples, re-solved as the stream grows
  const int W = 40; bool okc = true; int locks = 0;
  for (int end = 20; end <= N; end += 10) {
    Figure8 win; int from = end - W < 0 ? 0 : end - W;
    for (int i = from; i < end; i++) win.s.push_back(stream[i]);
    double yc[7]; solve8(win, yc);
    double ey = std::fabs(wrap(yc[0]-yawStar));
    double et = std::sqrt((yc[1]-tStar.x)*(yc[1]-tStar.x)+(yc[2]-tStar.y)*(yc[2]-tStar.y)+(yc[3]-tStar.z)*(yc[3]-tStar.z));
    bool good = ey<1e-2 && et<1e-2; okc = okc && (end<40 || good); if (good) locks++;
    std::printf("  continuous @%d samples (win %zu): yawerr=%.2e terr=%.2e %s\n", end, win.s.size(), ey, et, good?"locked":"settling");
  }

  bool ok = ok1 && okc;
  std::printf(ok ? "FIGURE8 OK: paired figure-8 recovers (yaw,t,mount) and holds continuously\n"
                 : "FIGURE8 FAIL\n");
  return ok?0:5;
}
