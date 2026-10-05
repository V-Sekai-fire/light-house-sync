// WIP / PARKED: PairCalib::solve() (pair_calib.cpp) and the null HMD driver are not yet written.
// Continuous figure-8 paired calibration, driver-side. Feed it controller poses (headset space) and
// the held tracker's pose (lighthouse space) each tick; it keeps a rolling window, re-solves the
// figure-8 fit, and reports the transform + coverage/quality for the UX meters. A "null driver"
// (pair_calib_null_test) can drive it with synthetic pairs so the whole path runs without hardware.
#pragma once
#include "align_figure8.h"
#include <deque>

namespace qlhs::align {

struct PairStatus {
  bool paired = false;      // residual low AND coverage full
  double residual_m = 1e9;  // RMS position error over the window (m)
  double travel = 0;        // translation-coverage meter 0..1 (pins yaw+t)
  double turn = 0;          // rotation-coverage meter 0..1 (pins the mount offset)
  int samples = 0;
  double yaw = 0; V3 t; V3 mount;
};

class PairCalib {
public:
  explicit PairCalib(size_t window = 60) : window_(window) {}

  // one tick: controller position (headset space) + held tracker pose (lighthouse space)
  void ingest(V3 ctrlPosHs, V3 trkPosLh, const M3 &trkRotLh) {
    win_.push_back({ctrlPosHs, trkPosLh, trkRotLh});
    while (win_.size() > window_) win_.pop_front();
  }
  void reset() { win_.clear(); have_ = false; }
  size_t size() const { return win_.size(); }

  // re-solve the current window; updates and returns status. Call ~1-5 Hz.
  PairStatus solve();

  // the current lighthouse->headset transform (valid once paired); applied to lighthouse devices.
  bool transform(double &yaw, V3 &t) const { if (!have_) return false; yaw = y_[0]; t = {y_[1], y_[2], y_[3]}; return true; }

  // thresholds (tunable)
  double lockResidualM = 0.01;  // 1 cm RMS
  double coverNeed = 0.8;       // both meters must reach this to pair
  double travelFull = 0.6;      // m of bbox diagonal = full travel meter
  double turnFull = 1.0;        // rad of rotation spread = full turn meter

private:
  size_t window_;
  std::deque<Sample8> win_;
  double y_[7] = {0,0,0,0,0,0,0};
  bool have_ = false;
};

} // namespace qlhs::align
