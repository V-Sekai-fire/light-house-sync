// Alignment residual on the GPU: the soft_l1 cost of sync.cpp Fit, summed over all sightings in
// df32 (double-single), for 9 variants of y = (yaw,tx,ty,tz) at once — base + central-difference
// +/-h on each of the 4 DOF. One workgroup, grid-strided; outputs 9 df32 (hi,lo) pairs. The host
// reads f = pair0 and g[j] = (pair[2j+1]-pair[2j+2])/(2h) in double. df32 makes the ~1e-? diffs
// resolvable in float32 arithmetic (plain float would drown them in rounding).

struct AlignParams {
  float4 y;   // yaw, tx, ty, tz
  float h;    // central-difference step
  uint n;     // sighting count
  uint pad0;
  uint pad1;
};

groupshared float s_hi[256];
groupshared float s_lo[256];

[[vk::binding(0, 0)]] ConstantBuffer<AlignParams> params;
[[vk::binding(1, 0)]] StructuredBuffer<float4> S;    // station positions (reference frame), .xyz
[[vk::binding(2, 0)]] StructuredBuffer<uint>   rk;   // sighting -> station index
[[vk::binding(3, 0)]] StructuredBuffer<float4> ro;   // sighting ray origin, .xyz
[[vk::binding(4, 0)]] StructuredBuffer<float4> rd;   // sighting ray dir (unit), .xyz
[[vk::binding(5, 0)]] StructuredBuffer<float>  rsw;  // sighting weight
[[vk::binding(6, 0)]] RWStructuredBuffer<float> dst; // 18 floats: 9 (hi,lo) pairs

void two_sum(float a, float b, out float hi, out float lo) {
  float h = a + b; float bb = h - a; float ah = h - bb; hi = h; lo = (a - ah) + (b - bb);
}
void quick_two_sum(float a, float b, out float hi, out float lo) {
  float h = a + b; float t = h - a; hi = h; lo = b - t;
}
void df_add(float xh, float xl, float yh, float yl, out float zh, out float zl) {
  float sh, sl; two_sum(xh, yh, sh, sl); float sl2 = sl + (xl + yl); quick_two_sum(sh, sl2, zh, zl);
}

float3 ryMul(float cy, float sy, float3 v) { return float3(cy * v.x + sy * v.z, v.y, -sy * v.x + cy * v.z); }

float sightCost(float3 Sk, float3 o, float3 d, float sw, float cy, float sy, float3 t) {
  float3 U = ryMul(cy, sy, Sk) + t - o;
  U = U * rsqrt(dot(U, U));
  float3 e = cross(U, d) * sw;
  const float F = 0.003;
  float cst = 0.0;
  [unroll] for (int c = 0; c < 3; c++) { float z = e[c] / F; cst += 2.0 * F * F * (sqrt(1.0 + z * z) - 1.0); }
  return cst;
}

[shader("compute")] [numthreads(256, 1, 1)]
void main(uint3 tid : SV_GroupThreadID) {
  uint t = tid.x;
  float h = params.h;
  float yaw = params.y.x; float3 tv = params.y.yzw;
  // 9 variants: 0 base; 1..2 yaw +/-h; 3..4 tx; 5..6 ty; 7..8 tz
  float cyv[9], syv[9]; float3 tvv[9];
  float c0 = cos(yaw), s0 = sin(yaw);
  cyv[0]=c0; syv[0]=s0; tvv[0]=tv;
  cyv[1]=cos(yaw+h); syv[1]=sin(yaw+h); tvv[1]=tv;
  cyv[2]=cos(yaw-h); syv[2]=sin(yaw-h); tvv[2]=tv;
  cyv[3]=c0; syv[3]=s0; tvv[3]=tv+float3(h,0,0);
  cyv[4]=c0; syv[4]=s0; tvv[4]=tv-float3(h,0,0);
  cyv[5]=c0; syv[5]=s0; tvv[5]=tv+float3(0,h,0);
  cyv[6]=c0; syv[6]=s0; tvv[6]=tv-float3(0,h,0);
  cyv[7]=c0; syv[7]=s0; tvv[7]=tv+float3(0,0,h);
  cyv[8]=c0; syv[8]=s0; tvv[8]=tv-float3(0,0,h);

  float ah[9], al[9];
  [unroll] for (int v = 0; v < 9; v++) { ah[v] = 0.0; al[v] = 0.0; }

  for (uint i = t; i < params.n; i += 256u) {
    float3 Sk = S[rk[i]].xyz; float3 o = ro[i].xyz; float3 d = rd[i].xyz; float sw = rsw[i];
    [unroll] for (int v = 0; v < 9; v++) {
      float cst = sightCost(Sk, o, d, sw, cyv[v], syv[v], tvv[v]);
      float nh, nl; df_add(ah[v], al[v], cst, 0.0, nh, nl); ah[v] = nh; al[v] = nl;
    }
  }

  // reduce each of the 9 accumulators, reusing one groupshared pair
  [unroll] for (int v = 0; v < 9; v++) {
    s_hi[t] = ah[v]; s_lo[t] = al[v];
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 128u; step > 0u; step >>= 1u) {
      if (t < step) { float nh, nl; df_add(s_hi[t], s_lo[t], s_hi[t+step], s_lo[t+step], nh, nl); s_hi[t]=nh; s_lo[t]=nl; }
      GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) { dst[2*v] = s_hi[0]; dst[2*v+1] = s_lo[0]; }
    GroupMemoryBarrierWithGroupSync();
  }
}
