#!/usr/bin/env python3
"""
Turn GPGPU-Sim logs from run_sim.sh into a markdown table.

    python3 sim/parse_sim.py sim/out/SM7_QV100/N256

Each log holds exactly one launch of one of our kernels, so the
last value printed for each statistic belongs to that kernel.

DRAM traffic is not reported: the tested configs set
-gpgpu_perf_sim_memcpy 1, which preloads cudaMemcpy'd data into L2,
and at the N we can afford to simulate the matrices fit in L2.
L2 accesses (global loads/stores that missed L1) are the useful
traffic measure here.
"""

import re
import sys
from pathlib import Path

# Core clock of the simulated GPU in MHz, keyed by config name.
# From -gpgpu_clock_domains in each config's gpgpusim.config.
CORE_MHZ = {
    "SM7_QV100": 1132.0,
    "SM75_RTX2060": 1365.0,
    "SM86_RTX3070": 1132.0,
}

ORDER = [
    "naive_uncoalesced", "naive",
    "smem8", "smem16", "smem32",
    "smem_rt8", "smem_rt16", "smem_rt32",
    "regtile",
]


def last(text, key, cast=float):
    """Last `key = value` in the log, or None if the simulator didn't print it."""
    hits = re.findall(rf"^\s*{re.escape(key)}\s*=\s*([-0-9.eE+]+)", text, re.M)
    return cast(hits[-1]) if hits else None


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main():
    out_dir = Path(sys.argv[1])
    config = out_dir.parent.name
    n = int(out_dir.name.lstrip("N"))
    mhz = CORE_MHZ.get(config)
    flops = 2.0 * n ** 3

    logs = {p.stem: p.read_text(errors="replace") for p in out_dir.glob("*.log")}
    names = [k for k in ORDER if k in logs] + sorted(k for k in logs if k not in ORDER)

    print(f"Simulated GPU: {config}  core clock: {mhz} MHz  N = {n}\n")
    print("| kernel | cycles | sim time (us) | GFLOP/s | IPC | occupancy % "
          "| L1D miss % | L2 accesses | L2 miss % | smem bank conflicts | check | wall (min) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")

    for k in names:
        t = logs[k]
        cycles = last(t, "gpu_sim_cycle", int)
        us = cycles / mhz if (cycles and mhz) else None
        gflops = flops / (us * 1e-6) / 1e9 if us else None
        l1 = last(t, "L1D_total_cache_miss_rate")
        l2 = last(t, "L2_total_cache_miss_rate")

        check = "ok" if re.search(rf"^{re.escape(k)}\s.*\sok$", t, re.M) else (
            "FAIL" if "FAIL" in t else "?")
        wall = last(t, "wall_seconds", int)

        print(f"| {k} | {fmt(cycles, ',')} | {fmt(us, '.1f')} | {fmt(gflops, '.0f')} "
              f"| {fmt(last(t, 'gpu_ipc'), '.1f')} | {fmt(last(t, 'gpu_occupancy'), '.1f')} "
              f"| {fmt(l1 * 100 if l1 is not None else None, '.1f')} "
              f"| {fmt(last(t, 'L2_total_cache_accesses', int), ',')} "
              f"| {fmt(l2 * 100 if l2 is not None else None, '.1f')} "
              f"| {fmt(last(t, 'gpgpu_n_shmem_bkconflict', int), ',')} "
              f"| {check} | {fmt(wall / 60 if wall else None, '.1f')} |")


if __name__ == "__main__":
    main()
