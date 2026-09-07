# VRAM Budget

The 4096 MiB ceiling is the whole design constraint. This file holds the
*measured* baseline (M0) and the projected budget (from `PLAN.md`).

## Measured baseline (M0)

Regenerate on any machine with:

```sh
./build/relwithdebinfo/bin/llm-devinfo --out docs/vram_baseline.log
```

`llm-devinfo` reports two different numbers, and the difference matters:

- **context overhead delta** — free VRAM before our first `cudaMalloc` minus free
  VRAM after. This is what the CUDA context itself costs us. Note that
  `cudaMemGetInfo` needs a context to be callable at all, so the driver has
  already done part of its setup before the "before" reading; this delta is a
  lower bound on true context cost, not the whole of it.
- **total reserved** — `total - free_after_context`. Includes the context *and*
  every other VRAM consumer on the machine (desktop compositor, browser, another
  CUDA process). This is the honest number to budget against: `free_after_context`
  is what we can actually allocate today.

On a laptop GPU that also drives a display, `total reserved` moves between runs.
Take the baseline with a quiet desktop, and re-take it before trusting any
allocator high-water-mark comparison in M5.

| Date | Machine | Total | Free post-context | Ctx delta | Reserved |
|---|---|---|---|---|---|
| _(pending: run `llm-devinfo` once the CUDA toolkit is installed)_ | | | | | |

## Projected budget

See the table in `PLAN.md`. Once the measured `free_after_context` above is
filled in, that number — not the assumed 200–400 MB — is what the allocator
sizing in M5 must start from.
