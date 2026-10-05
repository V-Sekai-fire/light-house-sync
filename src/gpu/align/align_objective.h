// Standalone alignment objective — a faithful port of QuestLHSync sync.cpp Fit's residual,
// decoupled from the driver so it can be validated on synthetic sightings before wiring in.
//
// Model (4 DOF, gravity-aligned): p_quest = Ry(yaw)*p_reference + t.
// For a sighting of station k from camera origin o along unit ray d:
//   U = Ry(yaw)*S[k] + t - o ;  e = (U/|U|) x d * sw          (3 residual comps)
// Robust cost: soft_l1, cost = sum 2*F^2*(sqrt(1+z^2)-1), z = e/F, F=FSCALE.
// Optional weak quadratic prior toward xa (as in Fit); off for pure-data recovery tests.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace qlhs::align {

struct V3 { double x = 0, y = 0, z = 0; };
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline double norm(V3 v) { return std::sqrt(dot(v, v)); }
// Rotation about +Y (gravity) applied to v.
inline V3 ryMul(double c, double s, V3 v) { return {c * v.x + s * v.z, v.y, -s * v.x + c * v.z}; }
inline V3 ry(double a, V3 v) { return ryMul(std::cos(a), std::sin(a), v); }
inline double wrap(double a) { const double P = 3.14159265358979323846; double r = std::fmod(a + P, 2 * P); if (r < 0) r += 2 * P; return r - P; }

// Fit constants (sync.h).
constexpr double FSCALE = 0.003;
constexpr double SIG_YAW = 0.5 * 3.14159265358979323846 / 180.0;  // 0.5 deg
constexpr double SIG_T = 0.05;                                     // m

struct Sighting { int k; V3 o, d; double sw = 1.0; };  // station index, origin, unit dir, weight

struct Problem {
  std::vector<V3> S;              // station positions in the reference frame
  std::vector<Sighting> rays;     // kept sightings (k in [0,S.size()))
  bool usePrior = false;          // weak quadratic prior toward xa
  double xa[4] = {0, 0, 0, 0};    // prior center (yaw,tx,ty,tz)

  // soft_l1 robust cost at y = {yaw, tx, ty, tz}
  double cost(const double *y) const {
    const double F = FSCALE, c = std::cos(y[0]), s = std::sin(y[0]);
    const V3 t{y[1], y[2], y[3]};
    double sum = 0.0;
    for (const Sighting &w : rays) {
      V3 U = ryMul(c, s, S[w.k]) + t - w.o;
      double nU = norm(U); if (nU > 0) U = {U.x / nU, U.y / nU, U.z / nU};
      V3 e = cross(U, w.d);
      double ex = e.x * w.sw, ey = e.y * w.sw, ez = e.z * w.sw;
      for (double rv : {ex, ey, ez}) { double z = rv / F; sum += 2 * F * F * (std::sqrt(1 + z * z) - 1); }
    }
    if (usePrior) {
      double a = F * wrap(y[0] - xa[0]) / SIG_YAW; sum += a * a;
      for (int i = 1; i < 4; i++) { double b = F * (y[i] - xa[i]) / SIG_T; sum += b * b; }
    }
    return sum;
  }

  // central-difference gradient (matches Fit's jac; h=1e-7)
  void grad(const double *y, double *g) const {
    double yp[4], ym[4];
    for (int j = 0; j < 4; j++) {
      for (int i = 0; i < 4; i++) { yp[i] = y[i]; ym[i] = y[i]; }
      const double h = 1e-7; yp[j] += h; ym[j] -= h;
      g[j] = (cost(yp) - cost(ym)) / (2 * h);
    }
  }
};

} // namespace qlhs::align
