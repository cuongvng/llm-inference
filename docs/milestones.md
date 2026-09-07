# Milestone Log

Status of each milestone from `PLAN.md`, with what was actually built and any
deviation from the plan.

## M0 — CMake + CUDA toolchain smoke test — **code complete, unverified**

**Done signal (PLAN.md):** `ctest` passes GPU-vs-CPU check; `cudaMemGetInfo`
baseline overhead recorded.

Built:

- `CMakeLists.txt` — C++17/CUDA17, `find_package(CUDAToolkit)`,
  `CMAKE_CUDA_ARCHITECTURES` from the `LLM_CUDA_ARCH` cache var (default `86`),
  `enable_testing()`.
- `CMakePresets.json` — `debug` (`-G -g`), `relwithdebinfo` (`-lineinfo`, the
  preset to profile with), `release`.
- `engine/kernels/vector_add.cu` — grid-stride vector add, the toolchain smoke
  test.
- `engine/src/device_info.cu` — device properties + `cudaMemGetInfo` before and
  after forcing context creation.
- `engine/tests/test_vector_add.cu` — ctest `test_vector_add`; GPU output must be
  bitwise equal to the CPU reference over 1,000,003 elements (a deliberately
  non-block-aligned count).
- `engine/tests/test_device_baseline.cu` — ctest `test_device_baseline`; records
  the baseline and fails if the binary's build arch doesn't match the running
  device's compute capability.
- `bench/src/devinfo_main.cc` — `llm-devinfo` CLI.

No test framework, by the dependency policy in `DESIGN_QA.md`: a test is a
`main()` returning non-zero.

**Blocked on:** the build machine (WSL2 Ubuntu 20.04) has no CUDA toolkit and no
CMake. GPU and driver are fine — `nvidia-smi` reports RTX 3050 Laptop, sm_86,
4096 MiB, driver 537.70 / CUDA 12.2 — so only the host-side toolchain is
missing. Run `scripts/setup_wsl_toolchain.sh` (needs sudo), then build and run
`ctest`. M0 is not done until both tests pass and the baseline row in
`docs/vram_budget.md` is filled in.

## M1 — Offline binary format + loader — not started

Open item carried from `PLAN.md`: the specific model (TinyLlama-1.1B vs
Qwen-0.5B/1.5B) is still unpicked, and it sets the vocab size in the header.
