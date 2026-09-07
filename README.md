# bare-cuda-4gib

A from-scratch CUDA LLM inference engine for a 4 GB card. Every kernel is
hand-written (CUDA Runtime API + nvcc, no cuBLAS/cuDNN/CUTLASS/Thrust) — the
point of the project is hands-on control of the GPU memory hierarchy under a
hard VRAM ceiling, not shipping a server.

- `PLAN.md` — architecture, VRAM budget, kernel inventory, milestones.
- `DESIGN_QA.md` — why those decisions were made, and what was rejected.
- `docs/milestones.md` — what's actually built so far.

Target GPU: RTX 3050 Laptop, compute capability 8.6, 4096 MiB.

## Prerequisites

- NVIDIA driver with CUDA 11.1+ (sm_86 support)
- CUDA toolkit 12.x (`nvcc`)
- CMake **3.18+** (`CMAKE_CUDA_ARCHITECTURES`)
- A C++17 host compiler

On WSL2 Ubuntu 20.04, the distro's own `nvidia-cuda-toolkit` (CUDA 10.1) and
`cmake` (3.16) are both too old. Install the upstream versions:

```sh
sudo bash scripts/setup_wsl_toolchain.sh
```

## Build and test

```sh
cmake --preset relwithdebinfo
cmake --build --preset relwithdebinfo
ctest --preset relwithdebinfo
```

Presets: `debug` (device `-G`), `relwithdebinfo` (`-lineinfo`, use this for
nsys/ncu), `release`. On a card other than sm_86, pass
`-DLLM_CUDA_ARCH=<cc>` — `test_device_baseline` fails loudly on a mismatch
rather than letting a JIT-compiled binary skew benchmarks.

## Device / VRAM baseline

```sh
./build/relwithdebinfo/bin/llm-devinfo --out docs/vram_baseline.log
```

Prints compute capability, SM count, shared memory per SM/block, L2 size,
theoretical bandwidth, and the `cudaMemGetInfo` VRAM baseline before and after
CUDA context creation. See `docs/vram_budget.md` for how to read those numbers.
