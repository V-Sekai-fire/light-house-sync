# L-BFGS-B kernels (Slang → C++)

These `*_emit.cpp` files are the **Slang → C++** emits (`slangc -target cpp`) of the
compact-form L-BFGS-B kernels from `V-Sekai-fire` `2-contract/lbfgsb`
(`kernels/drape/slang/*.slang`). They are reused here, not rewritten: the `.slang`
sources are the single source of truth and also emit SPIR-V (`-target spirv`) for the
GPU path. Each file inlines its own Slang C++ prelude, so it is self-contained.

Verified: all 11 compile cleanly with `zig cc -std=c++17 -O2 -c` (zig 0.17.0) — the
lhsync toolchain. Kernels:

    lb_box_project  lb_pg_inf_serial  lb_step_max_serial  lb_multi_dot_serial
    lb_compact  lb_ring_store  lb_apply_m  lb_cauchy  lb_subspace
    dot_reduce_serial  saxpby

The host L-BFGS-B reverse-communication loop (ported fresh, no guest/sandbox) drives
these as the vector ops; the alignment residual is a separate Slang kernel.

Upstream: `2-contract/lbfgsb` (Apache-2.0 OR MIT).
