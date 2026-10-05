// Reliability of acquisition: single-start (from yaw=0, like a cold LM) vs multi-start over a
// coarse yaw grid, both minimizing the full-window residual, over random ground truths. Shows
// multi-start locks reliably where single-start lands in the wrong basin (the yaw mirror / local
// minima) — the fix for "never locked reliably". The GPU makes the many starts cheap (benchmarked
// separately); here we measure the LOCK SUCCESS RATE of the algorithm.
// Build: zig c++ -std=c++17 -O2 align_acquire_test.cpp ../lbfgsb/lbfgsb.cpp ../lbfgsb/vec_cpu.cpp -o align_acquire_test.exe
#include "align_objective.h"
#include "../lbfgsb/lbfgsb.h"
#include "../lbfgsb/vec_cpu.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace qlhs::align;

// one L-BFGS-B solve of prob from a yaw seed; returns final cost, fills y[4]
static double solveFrom(const Problem& prob, double yawSeed, double* yout) {
  VecCpu vec; Lbfgsb solver(vec); LbfgsbParams p; p.max_iterations = 150;
  const float lo[4] = {-6.3f,-10,-10,-10}, hi[4] = {6.3f,10,10,10};
  float x0[4] = { (float)yawSeed, 0,0,0 };
  auto st = solver.start(4, x0, lo, hi, p);
  std::vector<float> xf; int guard=0;
  while (st!=Lbfgsb::CONVERGED && st!=Lbfgsb::FAIL && guard++<100000) {
    if (st==Lbfgsb::NEED_EVAL||st==Lbfgsb::TRY) { solver.readX(xf);
      double y[4]={xf[0],xf[1],xf[2],xf[3]}; double f=prob.cost(y),g[4]; prob.grad(y,g);
      float gf[4]={(float)g[0],(float)g[1],(float)g[2],(float)g[3]}; solver.setGradient(gf); st=solver.next(f);
    } else st=solver.next();
  }
  solver.readX(xf); for(int j=0;j<4;j++) yout[j]=xf[j];
  double y[4]={yout[0],yout[1],yout[2],yout[3]}; return prob.cost(y);
}

static double frand(double a,double b){ return a+(b-a)*(rand()/(double)RAND_MAX); }

int main(int argc,char**argv){
  const int TRIALS = argc>1?atoi(argv[1]):200;
  const int SEEDS = 12;                       // yaw grid: every 30 deg
  srand(7);
  Problem base; base.S = {{0,2,0},{3,2,-1},{-2,2,2}};

  int okSingle=0, okMulti=0;
  for(int tr=0; tr<TRIALS; tr++){
    double yawStar = frand(-3.14159, 3.14159);
    V3 tStar{ frand(-2,2), frand(-1.5,1.5), frand(-2,2) };
    double c=cos(yawStar), s=sin(yawStar);
    Problem prob; prob.S = base.S;
    // window: each sighting sees ONE station from a random camera (stations never all seen at once)
    for(int i=0;i<30;i++){ int k=rand()%3; V3 o{frand(-0.6,0.6),frand(1.4,1.8),frand(-0.6,0.6)};
      V3 Sq=ryMul(c,s,prob.S[k])+tStar; V3 d=Sq-o; double nd=norm(d); d={d.x/nd,d.y/nd,d.z/nd}; prob.rays.push_back({k,o,d,1.0}); }

    auto recovered=[&](const double* y){ double ey=std::fabs(wrap(y[0]-yawStar));
      double et=std::sqrt((y[1]-tStar.x)*(y[1]-tStar.x)+(y[2]-tStar.y)*(y[2]-tStar.y)+(y[3]-tStar.z)*(y[3]-tStar.z));
      return ey<1e-2 && et<1e-2; };

    // single-start (cold, yaw=0)
    double ys[4]; solveFrom(prob, 0.0, ys); if(recovered(ys)) okSingle++;

    // multi-start over the yaw grid, pick lowest cost
    double best=1e300, ybest[4]={0,0,0,0};
    for(int sdx=0; sdx<SEEDS; sdx++){ double seed=-3.14159 + (2*3.14159)*sdx/SEEDS; double y[4]; double cst=solveFrom(prob, seed, y);
      if(cst<best){ best=cst; for(int j=0;j<4;j++) ybest[j]=y[j]; } }
    if(recovered(ybest)) okMulti++;
  }
  printf("trials=%d  seeds=%d\n", TRIALS, SEEDS);
  printf("single-start (cold yaw=0) lock rate: %d/%d = %.1f%%\n", okSingle, TRIALS, 100.0*okSingle/TRIALS);
  printf("multi-start  (yaw grid)   lock rate: %d/%d = %.1f%%\n", okMulti,  TRIALS, 100.0*okMulti/TRIALS);
  bool ok = okMulti >= okSingle && okMulti >= (int)(0.98*TRIALS);
  printf(ok ? "ACQUIRE OK: multi-start locks reliably\n" : "ACQUIRE: see rates\n");
  return ok?0:5;
}
