// Figure-8 paired calibration: hold a Lighthouse tracker against a Frame controller and wave a
// figure-8. Each tick pairs the controller pose (headset space) with the tracker pose (Lighthouse
// space). We solve the gravity-aligned transform X = (yaw, t) AND the constant mount offset m
// (tracker frame) that the figure-8's rotation variety makes observable:
//     a_i = Ry(yaw) * (b_i + Rb_i * m) + t
// a_i: controller position (headset space); b_i, Rb_i: tracker position + orientation (Lighthouse).
// 7 unknowns (yaw, t3, m3), least-squares via the ported L-BFGS-B. No ML, no head-mounted tracker.
#pragma once
#include "align_objective.h"
#include <vector>
#include <cmath>

namespace qlhs::align {

struct M3 { double m[9]; };  // row-major
inline V3 m3mul(const M3 &R, V3 v) {
  return { R.m[0]*v.x + R.m[1]*v.y + R.m[2]*v.z,
           R.m[3]*v.x + R.m[4]*v.y + R.m[5]*v.z,
           R.m[6]*v.x + R.m[7]*v.y + R.m[8]*v.z };
}
inline M3 m3matmul(const M3 &A, const M3 &B) {
  M3 C;
  for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++)
    C.m[r*3+c] = A.m[r*3]*B.m[c] + A.m[r*3+1]*B.m[3+c] + A.m[r*3+2]*B.m[6+c];
  return C;
}
inline M3 rotX(double a){ double c=std::cos(a),s=std::sin(a); return {{1,0,0, 0,c,-s, 0,s,c}}; }
inline M3 rotY(double a){ double c=std::cos(a),s=std::sin(a); return {{c,0,s, 0,1,0, -s,0,c}}; }
inline M3 rotZ(double a){ double c=std::cos(a),s=std::sin(a); return {{c,-s,0, s,c,0, 0,0,1}}; }

struct Sample8 { V3 a; V3 b; M3 Rb; };  // controller pos (headset), tracker pos + orient (lighthouse)

struct Figure8 {
  std::vector<Sample8> s;

  // y = {yaw, tx,ty,tz, mx,my,mz}
  double cost(const double *y) const {
    double cy = std::cos(y[0]), sy = std::sin(y[0]);
    V3 t{y[1], y[2], y[3]}, m{y[4], y[5], y[6]};
    double sum = 0;
    for (const Sample8 &q : s) {
      V3 pred = ryMul(cy, sy, q.b + m3mul(q.Rb, m)) + t;
      V3 e = q.a - pred;
      sum += dot(e, e);
    }
    return sum;
  }
  void grad(const double *y, double *g) const {
    double yp[7], ym[7];
    for (int j = 0; j < 7; j++) {
      for (int i = 0; i < 7; i++) { yp[i] = y[i]; ym[i] = y[i]; }
      const double h = 1e-6; yp[j] += h; ym[j] -= h;
      g[j] = (cost(yp) - cost(ym)) / (2 * h);
    }
  }
};

} // namespace qlhs::align
