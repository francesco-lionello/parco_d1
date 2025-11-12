# plot_scaling.py
# Parse results.txt from spmv.
#
# Usage:
#   python3 plot_scaling.py results.txt
#
# Outputs (PNG files):
#   plot_seq_vs_parallel.png
#   plot_speedup.png
#   plot_scheduling.png
#   plot_bandwidth.png
#
# Requirements: pandas, matplotlib  (pip install pandas matplotlib)

import sys, re, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _legend_bold(ax):
    leg = ax.legend()
    if leg:
        for t in leg.get_texts():
            t.set_fontweight("bold")

def _apply_grid(ax):
    ax.grid(True, alpha=0.25, linewidth=0.6)

def _order_by_nnz(matrices_dict, names):
    # return name sorted by nnz
    return sorted(names, key=lambda n: matrices_dict[n]["meta"]["nnz"])


def parse_results(txt: str):
    matrices = {}
    current_matrix = None

    re_matrix_header = re.compile(
        r"Matrix Market:\s*(?P<path>\S+)\s*\|\s*M=(?P<M>\d+)\s*N=(?P<N>\d+)\s*nnz=(?P<nnz>\d+)\s*iters=(?P<iters>\d+)"
    )
    re_summary_line = re.compile(
        r"^(CSR seq|CSR omp\+simd|CSR omp)\s*:\s*best=\s*(?P<best>[0-9.]+)\s*ms\s*\|\s*p90=\s*(?P<p90>[0-9.]+)\s*ms\s*\|\s*GB/s=\s*(?P<gbps>[0-9.]+)"
    )
    re_scaling_header = re.compile(r"^---\s*(?P<ker>CSR omp\+simd|CSR omp)\s*scaling\s*\(OpenMP\)\s*---")
    re_schedule_header = re.compile(r"^===\s*schedule:\s*(?P<schedule>\w+)\s*===")
    re_thread_row = re.compile(
        r"^\s*(?P<thr>\d+)\s+(?P<best>[0-9.]+)\s+(?P<p90>[0-9.]+)\s+(?P<gbps>[0-9.]+)\s+(?P<speedup>[0-9.]+)\s+(?P<eff>[0-9.]+)"
    )

    lines = txt.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        m = re_matrix_header.search(line)
        if m:
            path = m.group("path")
            name = Path(path).name
            matrices[name] = {
                "meta": {
                    "path": path,
                    "M": int(m.group("M")),
                    "N": int(m.group("N")),
                    "nnz": int(m.group("nnz")),
                    "iters": int(m.group("iters")),
                },
                "summary": {},   # seq / omp / omp_simd
                "scaling": {},   # kernel -> schedule -> [rows]
            }
            current_matrix = name
            i += 1
            continue

        m2 = re_summary_line.search(line)
        if m2 and current_matrix:
            if line.startswith("CSR seq"):
                ker = "seq"
            elif line.startswith("CSR omp+simd"):
                ker = "omp_simd"
            elif line.startswith("CSR omp"):
                ker = "omp"
            else:
                ker = "unknown"
            matrices[current_matrix]["summary"][ker] = {
                "best_ms": float(m2.group("best")),
                "p90_ms": float(m2.group("p90")),
                "gbps": float(m2.group("gbps")),
            }
            i += 1
            continue

        m3 = re_scaling_header.search(line)
        if m3 and current_matrix:
            ker_full = m3.group("ker").strip()
            ker = "omp" if ker_full == "CSR omp" else "omp_simd"
            matrices[current_matrix]["scaling"].setdefault(ker, {})
            i += 1
            # inside scaling: multiple schedule blocks
            while i < len(lines):
                line2 = lines[i]
                msch = re_schedule_header.search(line2)
                if msch:
                    schedule = msch.group("schedule")
                    matrices[current_matrix]["scaling"][ker].setdefault(schedule, [])
                    i += 1
                    # skip header row if present
                    if i < len(lines) and ("thr" in lines[i] and "GB/s" in lines[i]):
                        i += 1
                    # read numeric rows
                    while i < len(lines):
                        row = lines[i]
                        if (re_matrix_header.search(row) or
                            re_scaling_header.search(row) or
                            re_schedule_header.search(row)):
                            break
                        mrow = re_thread_row.search(row)
                        if mrow:
                            matrices[current_matrix]["scaling"][ker][schedule].append({
                                "threads": int(mrow.group("thr")),
                                "best_ms": float(mrow.group("best")),
                                "p90_ms": float(mrow.group("p90")),
                                "gbps": float(mrow.group("gbps")),
                                "speedup": float(mrow.group("speedup")),
                                "eff": float(mrow.group("eff")),
                            })
                        i += 1
                    continue
                else:
                    if re_matrix_header.search(line2) or re_scaling_header.search(line2):
                        break
                    i += 1
            continue

        i += 1

    return matrices

def pick_p90_for_kernel(data, kernel_key):
    if kernel_key in data["summary"]:
        return data["summary"][kernel_key]["p90_ms"]
    scl = data["scaling"].get("omp" if kernel_key == "omp" else "omp_simd", {})
    st = scl.get("static", [])
    if st:
        st_sorted = sorted(st, key=lambda r: r["threads"])
        return st_sorted[-1]["p90_ms"]
    return float("nan")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 make_plots_from_results_txt.py results.txt")
        sys.exit(1)
    inpath = Path(sys.argv[1])
    if not inpath.exists():
        print(f"File not found: {inpath}")
        sys.exit(1)

    txt = inpath.read_text(encoding="utf-8", errors="ignore")
    matrices = parse_results(txt)

    # --- Meta table (per controllo) ---
    meta_rows = []
    for name, data in matrices.items():
        M = data["meta"]["M"]; N = data["meta"]["N"]; nnz = data["meta"]["nnz"]
        dens = 100.0 * nnz / (M*N) if M and N else float("nan")
        meta_rows.append({"matrix": name, "M": M, "N": N, "nnz": nnz, "density_pct": dens})
    meta_df = pd.DataFrame(meta_rows).sort_values("matrix")
    meta_df.to_csv("results.csv", index=False)

    # --- Plot 1: Sequential vs Parallel (p90) ---
    # --- Figure 1: Sequential vs OpenMP vs OpenMP+SIMD (p90 ms, log) ---
    mat_names = []
    seq_ms, omp_ms, simd_ms = [], [], []
    for name, data in matrices.items():
        mat_names.append(name)
        seq_ms.append(pick_p90_for_kernel(data, "seq"))
        omp_ms.append(pick_p90_for_kernel(data, "omp"))
        simd_ms.append(pick_p90_for_kernel(data, "omp_simd"))

    # ordina per nnz crescente
    mat_names = _order_by_nnz(matrices, mat_names)
    seq_ms  = [seq_ms[[n for n,_ in enumerate(mat_names)][i]] for i in range(len(mat_names))]  # safe reindex (no-op)
    # re-build by dict per sicurezza
    seq_ms  = [matrices[n]["summary"]["seq"]["p90_ms"]          if "seq" in matrices[n]["summary"] else float("nan") for n in mat_names]
    omp_ms  = [pick_p90_for_kernel(matrices[n], "omp") for n in mat_names]
    simd_ms = [pick_p90_for_kernel(matrices[n], "omp_simd") for n in mat_names]

    fig, ax = plt.subplots(figsize=(9,5))
    x = np.arange(len(mat_names))
    w = 0.28
    ax.set_yscale("log")
    ax.bar(x - w, seq_ms,  w, label="Sequential (p90 ms)")
    ax.bar(x,      omp_ms, w, label="OpenMP (p90 ms)")
    ax.bar(x + w, simd_ms, w, label="OpenMP+SIMD (p90 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(mat_names, rotation=30, ha="right")
    ax.set_ylabel("Time (ms, p90) — lower is better")
    ax.set_xlabel("Matrix (ordered by nnz)")
    ax.set_title("Figure 1 — Sequential vs Parallel execution times (p90 ms)")
    _apply_grid(ax)
    _legend_bold(ax)
    plt.tight_layout()
    plt.savefig("plot_seq_vs_parallel.png", dpi=200)
    plt.close(fig)


    # --- Plot 2: Speedup vs Threads (static) ---
    # --- Figure 2: Speedup vs Threads (OpenMP static, fixed ticks) ---
    THREAD_SET = [1,2,4,8,16,32]
    fig, ax = plt.subplots(figsize=(9,5))

    for name, data in matrices.items():
        seq_p90 = data["summary"].get("seq", {}).get("p90_ms", float("nan"))
        st = data["scaling"].get("omp", {}).get("static", [])
        if math.isnan(seq_p90) or seq_p90 <= 0 or not st:
            continue
        t2p90 = {r["threads"]: r["p90_ms"] for r in st}
        y = []
        for T in THREAD_SET:
            p90 = t2p90.get(T, float("nan"))
            y.append(seq_p90 / p90 if (p90 and p90>0 and not math.isnan(p90)) else np.nan)
        ax.plot(THREAD_SET, y, marker="o", label=name)

    # line of ideal scaling
    ax.plot([1, THREAD_SET[-1]], [1, THREAD_SET[-1]], linestyle="--", label="Ideal linear scaling")
    ax.set_xticks(THREAD_SET)
    ax.set_xlabel("Threads (static)")
    ax.set_ylabel("Speedup over sequential (p90 baseline)")
    ax.set_title("Figure 2 — Global Speedup vs Threads (OpenMP static)")
    _apply_grid(ax)
    _legend_bold(ax)
    plt.tight_layout()
    plt.savefig("plot_speedup.png", dpi=200)
    plt.close(fig)

    # --- Plot 3: Scheduling policies (fixed T) ---
    # --- Figure 3: Scheduling policies at fixed T ---
    counts = {}
    for data in matrices.values():
        for sched_rows in data["scaling"].get("omp", {}).values():
            for r in sched_rows:
                counts[r["threads"]] = counts.get(r["threads"],0)+1
    fixed_T = 16 if 16 in counts else max(counts, key=lambda t:(counts[t], -abs(t-16)))

    scheds = ["static","dynamic","guided"]
    rows = []
    for name, data in matrices.items():
        row = {"matrix": name}
        for s in scheds:
            val = float("nan")
            for r in data["scaling"].get("omp", {}).get(s, []):
                if r["threads"] == fixed_T:
                    val = r["p90_ms"]; break
            row[s] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    # order by nnz
    order = _order_by_nnz(matrices, df["matrix"].tolist())
    df = df.set_index("matrix").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(df))
    w = 0.22
    ax.set_yscale("log")
    for i, s in enumerate(scheds):
        ax.bar(x + (i-1)*w, df[s].values, w, label=s.capitalize())
    ax.set_xticks(x)
    ax.set_xticklabels(df["matrix"], rotation=30, ha="right")
    ax.set_ylabel("Time (ms, p90) — lower is better")
    ax.set_xlabel(f"Matrix (threads = {fixed_T})")
    ax.set_title("Figure 3 — OpenMP scheduling at fixed threads")
    _apply_grid(ax)
    _legend_bold(ax)
    plt.tight_layout()
    plt.savefig("plot_scheduling.png", dpi=200)
    plt.close(fig)


    # --- Plot 4: Bandwidth vs Threads (static) ---
    # --- Figure 4: Observed effective bandwidth vs Threads (all matrices, fixed ticks) ---
    THREAD_SET = [1,2,4,8,16,32]
    fig, ax = plt.subplots(figsize=(10,5))
    all_gbps = []

    for name, data in matrices.items():
        st = data["scaling"].get("omp", {}).get("static", [])
        if not st:
            continue
        t2bw = {r["threads"]: r["gbps"] for r in st}
        y = [t2bw.get(T, float("nan")) for T in THREAD_SET]
        all_gbps.extend([v for v in y if not math.isnan(v)])
        ax.plot(THREAD_SET, y, marker="o", label=name)

    if all_gbps:
        peak = max(all_gbps)
        ax.hlines(peak, xmin=THREAD_SET[0], xmax=THREAD_SET[-1], linestyles="--",
                  label=f"Peak observed ({peak:.1f} GB/s)")

    ax.set_xticks(THREAD_SET)
    ax.set_xlabel("Threads (static schedule)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("Figure 4 — Observed Bandwidth vs Threads (OpenMP static)")
    _apply_grid(ax)
    _legend_bold(ax)
    plt.tight_layout()
    plt.savefig("plot_bandwidth.png", dpi=200)   # tieni anche 'all'
    plt.close(fig)

    print("Done. Generated PNGs:")
    for fn in ["plot_seq_vs_parallel.png","plot_speedup.png","plot_scheduling.png","plot_bandwidth.png"]:
        print(" -", fn)

if __name__ == "__main__":
    main()
