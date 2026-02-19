#!/usr/bin/env python3
"""
weta_igloo_correction.py
========================
Step 4: Cold-trapping correction using the IGLOO null model
(Giraldo et al. 2019, Sci Rep 9:3974).

This version vectorizes the entire IGLOO random walk across all simulated
animals simultaneously, avoiding IGLOO's per-step np.vstack bottleneck.
Supports GPU acceleration via --use_gpu (requires cupy + CUDA).

Usage
-----
    python weta_igloo_correction.py \
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \
        --n_sim 10000 --walk_dur 3600 --sps 25

    # With GPU:
    python weta_igloo_correction.py \
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \
        --n_sim 10000 --walk_dur 3600 --use_gpu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
import json
import argparse
from collections import defaultdict
try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback if tqdm not installed
    class tqdm:
        def __init__(self, total=100, **kwargs):
            self.total = total
            self.n = 0
            self.desc = kwargs.get("desc", "")
        def update(self, n=1):
            self.n += n
            print(f"\r  {self.desc}: {self.n}/{self.total}%", end="", flush=True)
        def close(self):
            print()

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
SPECIES_COLOURS = {
    "H. maori":      "#D55E00",
    "H. crassidens": "#56B4E9",
    "H. thoracica":  "#009E73",
}
SPECIES_ORDER = ["H. crassidens", "H. maori", "H. thoracica"]


# ──────────────────────────────────────────────────────────────────────────────
# Backend: numpy or cupy
# ──────────────────────────────────────────────────────────────────────────────

def get_backend(use_gpu):
    """Return (xp, backend_name) — either cupy or numpy."""
    if use_gpu:
        try:
            import cupy as cp
            # Quick sanity check
            _ = cp.zeros(1)
            print("[INFO] Using GPU (cupy + CUDA)")
            return cp, "gpu"
        except Exception as e:
            print(f"[WARN] GPU requested but cupy unavailable: {e}")
            print("[INFO] Falling back to CPU (numpy)")
    return np, "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized IGLOO math (from locomotionInterpolation.py)
# ──────────────────────────────────────────────────────────────────────────────

def _gauss(xp, x, a, x0, sigma):
    return a * xp.exp(-((x - x0) / (2.0 * sigma)) ** 2)


def _poly2(xp, x, a, b, c):
    return a * x**2 + b * x + c


def _poly3(xp, x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def _calc_parameters(rearing_t):
    """Rearing-temperature-dependent velocity and duration parameters.
    Identical to locomotionInterpolation.calcParameters()."""
    T = rearing_t
    v = np.empty(3)
    v[0] = _poly2(np, T, 0.12099083, -5.66690323, 70.7646241)
    v[1] = _poly2(np, T, -3.74142886e-02, 2.11954454e+00, -4.34662354e+01)
    v[2] = _poly2(np, T, -0.07461945, 3.14576967, -22.92563255)

    d = np.empty(8)
    d[0] = 0.12916312 * T + (-2.812052)
    d[1] = _poly2(np, T, 0.14534352, -7.95321438, 110.60810592)
    d[2] = _poly2(np, T, 0.02662904, -1.28284224, 15.65186953)
    d[3] = _poly2(np, T, 11.66468748, -641.51376808, 8753.61384674)
    d[4] = _poly2(np, T, 0.14445866, -6.92317723, 75.43250874)
    d[5] = _poly2(np, T, -0.24815044, 11.88971369, -129.60720025)
    d[6] = _poly2(np, T, 0.15978689, -7.64871786, 83.63873179)
    d[7] = _poly2(np, T, -0.0612502, 2.92865988, -32.21078329)

    return v, d


def _vel_func(xp, p, tb, v):
    """Vectorized velocity function for adults."""
    temp_part = _gauss(xp, tb, 1.6, 34.0, 2.27) + _gauss(xp, tb, 1.8, 26.0, 5.41)
    prob_part = _poly2(xp, p, v[0], v[1], v[2])
    return temp_part * prob_part


def _dur_func(xp, p, tb, d):
    """Vectorized duration function for adults."""
    temp_part = _gauss(xp, tb, d[0], 34.0, d[1]) + _gauss(xp, tb, d[2], 18.0, d[3])
    prob_part = _poly3(xp, p, d[4], d[5], d[6], d[7])
    return temp_part * prob_part


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized heat conduction (from IGLOO.drosoTbyConduction)
# ──────────────────────────────────────────────────────────────────────────────

def _update_body_temp(xp, body_t, ambient_t, duration):
    """Vectorized Drosophila body temperature update.

    Conductance model: cylinder of water, r=0.5mm, l=2mm.
    conductance = 0.6 W/(m·K), surface = 7.85e-6 m², D = 1e-3 m
    1 J heats 1.57 mg water by 152.23 °C
    """
    dT = ambient_t - body_t
    Q = 0.6 * 7.85e-6 * (dT / 1e-3) * duration
    temp_change = Q * 152.23

    new_body_t = body_t + temp_change

    # Clamp: don't overshoot ambient temperature
    warming = ambient_t > body_t
    cooling = ambient_t < body_t
    overshoot_warm = warming & (new_body_t > ambient_t)
    overshoot_cool = cooling & (new_body_t < ambient_t)
    new_body_t = xp.where(overshoot_warm, ambient_t, new_body_t)
    new_body_t = xp.where(overshoot_cool, ambient_t, new_body_t)

    return new_body_t


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_null_vectorized(n_sim, gradient_ext, gradient_dist, walk_dur,
                              rearing_t, sps, use_gpu=False):
    """Simulate n_sim flies with no temperature preference in a gradient.

    All flies are updated simultaneously as arrays — no per-fly loops.

    Returns
    -------
    null_mean, null_sem, bins : as expected by correct_and_plot()
    """
    from scipy import stats as sp_stats

    xp, backend = get_backend(use_gpu)

    v_params, d_params = _calc_parameters(rearing_t)
    deg_per_mm = abs(gradient_ext[1] - gradient_ext[0]) / gradient_dist

    # ── Initial state for all flies ──
    position = xp.random.uniform(0, gradient_dist, size=n_sim)
    ambient_t = position * deg_per_mm + gradient_ext[0]
    body_t = ambient_t.copy()
    time_acc = xp.zeros(n_sim)  # accumulated time per fly
    alive = xp.ones(n_sim, dtype=bool)  # mask: still simulating

    # ── Pre-allocate histogram accumulators ──
    # We accumulate temperature histograms on-the-fly instead of storing
    # full traces (which would be n_sim × ~30000 steps = too much memory).
    # Strategy: at each step, add duration-weighted bin counts per fly.
    bin_num = int(gradient_ext[1] - gradient_ext[0])
    bins_np = np.linspace(gradient_ext[0] - 0.5, gradient_ext[1] + 0.5, bin_num + 2)

    # Per-fly histogram accumulators (always on CPU)
    hist_matrix = np.zeros((n_sim, bin_num + 1))

    # For progress tracking
    steps_done = 0
    pbar = tqdm(total=100, desc="IGLOO simulation", unit="%",
                bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]")

    while True:
        n_alive = int(xp.sum(alive))
        if n_alive == 0:
            break

        # Progress based on fraction of flies finished
        pct = int(100 * (1 - n_alive / n_sim))
        if pct > pbar.n:
            pbar.update(pct - pbar.n)

        # ── Generate random numbers for active flies ──
        p_vel = xp.random.random(n_sim)
        p_dur = xp.random.random(n_sim)
        direction = xp.where(xp.random.random(n_sim) < 0.5, -1.0, 1.0)

        # ── Compute velocity and duration from body temperature ──
        velocity = _vel_func(xp, p_vel, body_t, v_params)
        duration = _dur_func(xp, p_dur, body_t, d_params)

        # Clamp negatives (fit functions can go slightly negative at extremes)
        velocity = xp.maximum(velocity, 0.0)
        duration = xp.maximum(duration, 0.001)

        # ── Update time ──
        time_acc += duration * alive

        # ── Check which flies are now done ──
        newly_done = alive & (time_acc >= walk_dur)

        # ── Update position with reflective boundaries ──
        step = direction * velocity * duration * alive
        new_pos = position + step

        # Reflect from far wall
        overshoot_far = new_pos > gradient_dist
        new_pos = xp.where(overshoot_far, 2 * gradient_dist - new_pos, new_pos)

        # Reflect from near wall
        overshoot_near = new_pos < 0
        new_pos = xp.where(overshoot_near, -new_pos, new_pos)

        # Safety clamp (double reflection edge cases)
        new_pos = xp.clip(new_pos, 0, gradient_dist)

        position = new_pos

        # ── Update ambient temperature from position ──
        ambient_t = position * deg_per_mm + gradient_ext[0]

        # ── Update body temperature by conduction ──
        body_t = _update_body_temp(xp, body_t, ambient_t, duration)

        # ── Accumulate into histograms (vectorized, duration-weighted) ──
        if backend == "gpu":
            amb_cpu = xp.asnumpy(ambient_t)
            dur_cpu = xp.asnumpy(duration)
            alive_cpu = xp.asnumpy(alive)
        else:
            amb_cpu = ambient_t
            dur_cpu = duration
            alive_cpu = alive

        # Vectorized bin assignment for all flies at once
        bin_indices = np.searchsorted(bins_np, amb_cpu) - 1
        bin_indices = np.clip(bin_indices, 0, bin_num)

        # Use np.add.at for scatter-add (no Python loop over flies)
        fly_indices = np.where(alive_cpu)[0]
        np.add.at(hist_matrix, (fly_indices, bin_indices[fly_indices]),
                  dur_cpu[fly_indices])

        # ── Mark finished flies ──
        if backend == "gpu":
            alive = alive & ~newly_done
        else:
            alive[newly_done] = False

        steps_done += 1

    pbar.update(100 - pbar.n)
    pbar.close()
    print(f"  Completed in {steps_done} vectorized steps")

    # ── Normalise per-fly histograms ──
    row_sums = hist_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    hist_matrix = hist_matrix / row_sums

    null_mean = np.mean(hist_matrix, axis=0)
    null_sem = sp_stats.sem(hist_matrix, axis=0)

    total = null_mean.sum()
    if total > 0:
        null_mean /= total
        null_sem /= total

    return null_mean, null_sem, bins_np


# ──────────────────────────────────────────────────────────────────────────────
# Weta data loading
# ──────────────────────────────────────────────────────────────────────────────

def get_species(trial_id: str) -> str:
    if trial_id.startswith("hcrass"):
        return "H. crassidens"
    elif trial_id.startswith("hm"):
        return "H. maori"
    elif trial_id.startswith("hthora"):
        return "H. thoracica"
    return "unknown"


def load_weta_temperature_data(processed_dir: str):
    """Load all _animal_temperature.npy files grouped by species."""
    species_temps = defaultdict(list)
    all_tmin, all_tmax = np.inf, -np.inf

    files = sorted([f for f in os.listdir(processed_dir)
                    if f.endswith("_animal_temperature.npy")])

    for f in files:
        trial_id = f.replace("_animal_temperature.npy", "")
        sp = get_species(trial_id)
        temp = np.load(os.path.join(processed_dir, f))
        valid = temp[~np.isnan(temp)]
        if len(valid) == 0:
            continue
        species_temps[sp].append(valid)
        all_tmin = min(all_tmin, valid.min())
        all_tmax = max(all_tmax, valid.max())

    return species_temps, float(all_tmin), float(all_tmax)


def compute_observed_histogram(temp_arrays, bins):
    """Per-animal normalised histograms → mean ± 95% CI."""
    from scipy import stats

    n_animals = len(temp_arrays)
    n_bins = len(bins) - 1
    hist_matrix = np.zeros((n_animals, n_bins))

    for i, temps in enumerate(temp_arrays):
        h, _ = np.histogram(temps, bins=bins, density=True)
        h = h / h.sum() if h.sum() > 0 else h
        hist_matrix[i, :] = h

    mean_hist = np.mean(hist_matrix, axis=0)
    sem_hist = stats.sem(hist_matrix, axis=0)

    total = mean_hist.sum()
    if total > 0:
        mean_hist /= total
        sem_hist /= total

    return mean_hist, sem_hist


# ──────────────────────────────────────────────────────────────────────────────
# Correction and plotting
# ──────────────────────────────────────────────────────────────────────────────

def correct_and_plot(species_temps, null_mean, null_sem, bins, output_dir):
    """For each species: subtract null → preference index, plot, save CSV."""

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    csv_rows = []

    for sp in SPECIES_ORDER:
        if sp not in species_temps or len(species_temps[sp]) == 0:
            continue

        n_animals = len(species_temps[sp])
        obs_mean, obs_sem = compute_observed_histogram(species_temps[sp], bins)

        # Preference index
        pref_index = obs_mean - null_mean

        # 95% CI classification
        obs_lower = obs_mean - 1.96 * obs_sem
        obs_upper = obs_mean + 1.96 * obs_sem
        preferred = obs_lower > null_mean
        avoided = obs_upper < null_mean
        tolerable = ~preferred & ~avoided

        # Weighted mean preferred temperature
        if np.any(preferred):
            pref_temps = bin_centers[preferred]
            pref_weights = pref_index[preferred]
            tp_corrected = np.average(pref_temps, weights=pref_weights)
        else:
            tp_corrected = np.nan

        tp_uncorrected = np.average(bin_centers, weights=obs_mean)

        # Avoidance boundaries
        avoided_temps = bin_centers[avoided]
        if len(avoided_temps) > 0 and not np.isnan(tp_corrected):
            cold_side = avoided_temps[avoided_temps < tp_corrected]
            hot_side = avoided_temps[avoided_temps > tp_corrected]
            cold_avoidance_start = float(cold_side.max()) if len(cold_side) > 0 else np.nan
            hot_avoidance_start = float(hot_side.min()) if len(hot_side) > 0 else np.nan
        else:
            cold_avoidance_start = np.nan
            hot_avoidance_start = np.nan

        # Tolerable range
        tolerable_temps = bin_centers[tolerable | preferred]
        tol_range = (float(tolerable_temps.min()), float(tolerable_temps.max())) \
            if len(tolerable_temps) > 0 else (np.nan, np.nan)

        print(f"\n  {sp} (n={n_animals}):")
        print(f"    T_p uncorrected : {tp_uncorrected:.1f} °C")
        print(f"    T_p corrected   : {tp_corrected:.1f} °C")
        if not np.isnan(cold_avoidance_start):
            print(f"    Cold avoidance  : < {cold_avoidance_start:.1f} °C")
        if not np.isnan(hot_avoidance_start):
            print(f"    Hot avoidance   : > {hot_avoidance_start:.1f} °C")
        print(f"    Tolerable range : {tol_range[0]:.1f}–{tol_range[1]:.1f} °C")

        # CSV rows
        for j, tc in enumerate(bin_centers):
            cat = "preferred" if preferred[j] else ("avoided" if avoided[j] else "tolerable")
            csv_rows.append([
                sp, round(tc, 1),
                round(obs_mean[j], 6), round(obs_sem[j], 6),
                round(null_mean[j], 6), round(null_sem[j], 6),
                round(pref_index[j], 6), cat
            ])

        # ── Plot ──
        sp_label = sp.replace("H. ", "H_")
        col = SPECIES_COLOURS[sp]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                        gridspec_kw={"height_ratios": [1, 1]})

        # Top: observed vs null
        ax1.bar(bin_centers, obs_mean, width=np.diff(bins[:2])[0] * 0.85,
                alpha=0.6, color=col, label=f"{sp} observed (n={n_animals})")
        ax1.errorbar(bin_centers, obs_mean, yerr=1.96 * obs_sem,
                     fmt="none", ecolor="k", capsize=2, linewidth=0.8)
        ax1.step(np.append(bins, bins[-1]),
                 np.append(np.append([0], null_mean), 0),
                 where="pre", color="red", linewidth=1.5,
                 label="IGLOO null model")
        if not np.isnan(tp_corrected):
            ax1.axvline(tp_corrected, color="darkred", ls="--", lw=1,
                        label=f"$T_p$ corrected = {tp_corrected:.1f} °C")
        ax1.set_ylabel("Probability density")
        ax1.set_title(f"Temperature preference — {sp}")
        ax1.legend(fontsize=8)

        # Bottom: preference index
        for j, tc in enumerate(bin_centers):
            w = np.diff(bins[:2])[0] * 0.85
            if preferred[j]:
                ax2.bar(tc, pref_index[j], width=w, color="red", alpha=0.6)
            elif avoided[j]:
                ax2.bar(tc, pref_index[j], width=w, color="blue", alpha=0.6)
            else:
                ax2.bar(tc, pref_index[j], width=w, color="grey", alpha=0.4)
        ax2.errorbar(bin_centers, pref_index, yerr=1.96 * obs_sem,
                     fmt="none", ecolor="k", capsize=2, linewidth=0.8)
        ax2.axhline(0, color="k", linewidth=0.5)
        ax2.set_xlabel("Temperature (°C)")
        ax2.set_ylabel("Preference index\n(observed − null)")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red", alpha=0.6, label="Preferred"),
            Patch(facecolor="blue", alpha=0.6, label="Avoided"),
            Patch(facecolor="grey", alpha=0.4, label="Tolerable"),
        ]
        ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"igloo_correction_{sp_label}.png"),
                    dpi=200)
        fig.savefig(os.path.join(output_dir, f"igloo_correction_{sp_label}.svg"))
        plt.close(fig)
        print(f"    Plot → igloo_correction_{sp_label}.png/.svg")

    # Write CSV
    csv_path = os.path.join(output_dir, "igloo_correction_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "temperature_degC",
                     "observed_mean", "observed_sem",
                     "null_mean", "null_sem",
                     "preference_index", "classification"])
        w.writerows(csv_rows)
    print(f"\n  CSV → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: IGLOO cold-trapping correction for weta (vectorized)."
    )
    parser.add_argument("--base_dir", type=str,
                        default="/home/geuba03p/weta_project/weta_videos_cropped")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_sim", type=int, default=1000,
                        help="Number of simulated animals (default: 1000)")
    parser.add_argument("--walk_dur", type=float, default=600.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--rearing_t", type=float, default=10.0,
                        help="Rearing temperature for locomotion model")
    parser.add_argument("--sps", type=int, default=25,
                        help="Samples per second (for output resampling)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU via cupy (requires cupy + CUDA)")

    args = parser.parse_args()

    processed_dir = args.processed_dir or os.path.join(args.base_dir, "processed_trajectories")
    output_dir = args.output_dir or os.path.join(args.base_dir, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load weta data ──
    print("── Loading weta temperature data ──")
    species_temps, t_min, t_max = load_weta_temperature_data(processed_dir)

    t_min_round = np.floor(t_min)
    t_max_round = np.ceil(t_max)

    n_total = sum(len(v) for v in species_temps.values())
    print(f"  {n_total} trials across {len(species_temps)} species")
    print(f"  Temperature range: {t_min:.1f}–{t_max:.1f} °C "
          f"(bins: {t_min_round:.0f}–{t_max_round:.0f})")

    # ── 2. Vectorized IGLOO null model ──
    print("\n── Running vectorized IGLOO null model ──")
    null_mean, null_sem, bins = simulate_null_vectorized(
        n_sim=args.n_sim,
        gradient_ext=(t_min_round, t_max_round),
        gradient_dist=600.0,
        walk_dur=args.walk_dur,
        rearing_t=args.rearing_t,
        sps=args.sps,
        use_gpu=args.use_gpu,
    )
    print(f"  Null model: {len(null_mean)} bins")

    # ── 3. Correct and plot ──
    print("\n── Correcting for cold-trapping ──")
    correct_and_plot(species_temps, null_mean, null_sem, bins, output_dir)

    print(f"\n[DONE] All outputs in {output_dir}/")


if __name__ == "__main__":
    main()