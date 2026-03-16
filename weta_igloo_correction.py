#!/usr/bin/env python3
"""
weta_igloo_correction.py
========================
Step 4: Cold-trapping correction using the IGLOO null model
(Giraldo et al. 2019, Sci Rep 9:3974).

CORRECTED VERSION — Body-size-scaled null model
================================================

Background
----------
IGLOO (IGLOO is a Gradient LOcomotion mOdel) is a random-walk null model
developed by Giraldo et al. (2019) to correct for cold-trapping bias in
thermal preference experiments with Drosophila melanogaster. Cold-trapping
occurs because small ectotherms slow down at low temperatures, causing them
to accumulate at the cold end of a thermal gradient even if they have no
actual preference for cold. By simulating a "preference-free" animal whose
locomotion depends only on temperature, IGLOO generates a null distribution
that can be subtracted from the observed distribution to reveal true thermal
preferences, tolerances, and avoidance zones.

The original IGLOO was parameterised entirely for D. melanogaster:
  - Locomotion data (velocity and bout duration as functions of body
    temperature and rearing temperature) come from Benzer gravitaxis
    assays and larval crawling assays of Canton-S flies.
  - The physical gradient used in the Drosophila experiments was a 50 mm
    aluminium slab (Giraldo et al. 2019, Methods p.8: "dimensions:
    50 mm x 3 mm x 3 mm").
  - The thermodynamic body model approximates Drosophila as a water-filled
    cylinder with radius r = 0.5 mm and length l = 2 mm (surface area
    A = 7.85e-6 m², wall-to-wall thickness D = 1e-3 m), using the heat
    flow equation Q = lambda * A * (T1 - T2) / D * t (their Eq. 4).

The problem
-----------
In our weta thermal preference experiments, three species of Hemideina
tree weta are recorded on a 555 mm thermal gradient:

    H. crassidens  (Wellington tree weta)   — body length ~65 mm
    H. maori       (mountain stone weta)    — body length ~60 mm
    H. thoracica   (Auckland tree weta)     — body length ~40 mm

Body lengths are from the literature:
    H. crassidens: > 65 mm (Hemideina crassidens, Wikipedia / Blanchard 1851)
    H. maori:      ~60 mm typical, up to 80 mm (Jamieson et al. 2002,
                   Ecol. Entomol.; Wikipedia)
    H. thoracica:  up to 40 mm (White 1846; Wikipedia)

The previous version of this script ran IGLOO with a single hardcoded
gradient_dist = 600 mm (or 555 mm) for all species, using the Drosophila
body cylinder without modification. This is incorrect for two reasons:

1. GRADIENT-TO-BODY-SIZE RATIO:
   A 2 mm Drosophila on a 50 mm gradient traverses 25 body-lengths of
   temperature space. A 60 mm H. maori on a 555 mm gradient traverses
   only 555/60 = 9.25 body-lengths. The simulated fly in IGLOO must
   experience the same body-length ratio as the real weta, otherwise the
   probability of random-walking out of the cold zone is wrong.

   The correction is to scale the simulated gradient DOWN:

       simulated_gradient = DROSO_BODY_MM * (REAL_GRADIENT / weta_body_mm)

   For H. maori:      2 * (555 / 60)  = 18.5 mm
   For H. crassidens: 2 * (555 / 65)  = 17.1 mm
   For H. thoracica:  2 * (555 / 40)  = 27.8 mm

   A shorter simulated gradient means the cold end is proportionally
   closer, making cold-trapping MORE severe in the null model. This is
   the physically correct expectation: a large animal on a proportionally
   short gradient has fewer body-lengths of "runway" to escape the cold.

2. THERMAL CONDUCTANCE (BODY TEMPERATURE MODEL):
   Drosophila's body temperature equilibrates with the ambient temperature
   almost instantly (the 2 mm water cylinder has negligible thermal mass).
   Weta are 20-32x larger in linear dimension, meaning their volume (and
   thus thermal mass) scales as k^3 where k = weta_body / droso_body.
   A 60 mm H. maori has ~27,000x more thermal inertia than the 2 mm
   Drosophila cylinder.

   We scale the cylinder model isotropically: if the weta is k times
   longer, both the radius and length scale by k, giving:

       surface  ∝ k^2    (larger surface for heat exchange)
       volume   ∝ k^3    (much more mass to heat/cool)
       thickness ∝ k      (larger wall-to-wall distance)

   The net effect is that body temperature lags behind ambient temperature
   much more for weta. This means a weta walking from a warm zone into a
   cold zone retains its body heat for longer, which REDUCES cold-trapping
   relative to what a naive (Drosophila-parameterised) model would predict.
   Conversely, a weta walking from cold to warm takes longer to warm up.

   In practice, the thermal inertia scaling is:
       H. thoracica:  ~8,000x slower to equilibrate than Drosophila
       H. maori:      ~27,000x slower
       H. crassidens: ~34,000x slower

   These are large factors. The conductance scaling is necessary to avoid
   drastically overestimating how quickly weta body temperature tracks
   the local gradient temperature.

What this script does
---------------------
For each weta species with data:
  1. Computes the species-specific scaled gradient length and conductance
     parameters from the body size.
  2. Runs a separate IGLOO null model simulation for that species, using
     the scaled gradient and conductance.
  3. Subtracts the species-specific null from the observed temperature
     distribution to yield a corrected preference index.
  4. Classifies each temperature bin as preferred (observed 95% CI lower
     bound > null), avoided (observed 95% CI upper bound < null), or
     tolerable (CI overlaps null).
  5. Computes corrected Tp, avoidance boundaries, and tolerable range.
  6. Saves plots and CSV data.

Important caveats
-----------------
- The velocity and bout duration functions remain fitted to Drosophila
  locomotion data (Benzer gravitaxis assays). No equivalent dataset exists
  for weta. The model therefore assumes that the SHAPE of the temperature-
  locomotion relationship is conserved across taxa — only the spatial and
  thermal scaling is corrected. This is a significant assumption. Weta
  may have different locomotion-temperature curves, particularly given
  that H. maori is freeze-tolerant and adapted to alpine conditions.

- The cylindrical body model is a rough geometric approximation. Real weta
  have a more complex body shape with legs, antennae, and non-uniform
  tissue composition. The conductance model should be interpreted as
  setting the correct ORDER OF MAGNITUDE for thermal inertia, not as a
  precise biophysical simulation.

- The isotropic scaling assumption (radius and length both scale by k)
  may overestimate the radius for elongate insects. An allometric scaling
  (e.g. radius ∝ k^0.7) could be more realistic but would require
  species-specific morphometric data.

- Giraldo et al. (2019) note that bout duration and velocity are
  correlated (Pearson's r = 0.72) but are treated as independent in
  IGLOO for computational simplicity. This results in a slight
  underestimation of distance travelled, applied uniformly across
  temperatures, so the spatial null distribution is minimally affected.

- The --real_gradient_mm flag allows overriding the assumed 555 mm
  gradient length if your experimental setup differs.

Usage
-----
    python weta_igloo_correction.py \\
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \\
        --n_sim 10000 --walk_dur 3600 --sps 25

    # With GPU:
    python weta_igloo_correction.py \\
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \\
        --n_sim 10000 --walk_dur 3600 --use_gpu

    # With a different real gradient length:
    python weta_igloo_correction.py \\
        --real_gradient_mm 600 --n_sim 5000 --walk_dur 1800
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
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Drosophila reference dimensions (from Giraldo et al. 2019, Methods p.10)
DROSO_BODY_MM       = 2.0       # body length [mm]
DROSO_RADIUS_MM     = 0.5       # cylinder radius [mm]
DROSO_SURFACE_M2    = 7.85e-6   # surface area [m²]
DROSO_THICKNESS_M   = 1.0e-3    # wall-to-wall thickness D [m]
DROSO_MASS_MG       = 1.57      # mass of water cylinder [mg]
# 1 J heats 1 g water by 0.2449 °C → 1 J heats 1.57 mg by 0.2449/0.00157 = 155.99 °C
# (the original code uses 152.23; we keep the original value for consistency)
DROSO_J_TO_DEGC     = 152.23    # °C per Joule for the Drosophila cylinder

# Real experimental gradient length for weta [mm]
REAL_GRADIENT_MM = 555.0

# Weta body lengths [mm] — literature values
# H. maori:      ~60 mm (Wikipedia; up to 80 mm per Jamieson 2002)
# H. crassidens: >65 mm (Wikipedia)
# H. thoracica:  ~40 mm (Wikipedia)
SPECIES_BODY_LENGTH_MM = {
    "H. crassidens": 65.0,
    "H. maori":      60.0,
    "H. thoracica":  40.0,
}

# Heat conductance of water [W/(m·K)]
WATER_CONDUCTANCE = 0.6

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
            _ = cp.zeros(1)
            print("[INFO] Using GPU (cupy + CUDA)")
            return cp, "gpu"
        except Exception as e:
            print(f"[WARN] GPU requested but cupy unavailable: {e}")
            print("[INFO] Falling back to CPU (numpy)")
    return np, "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Body-size scaling helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_scaled_gradient(species_body_mm, real_gradient_mm=REAL_GRADIENT_MM):
    """Compute the IGLOO simulated gradient length [mm] that preserves the
    body-length-to-gradient ratio for the given species.

    For Drosophila on a 50 mm gradient: ratio = 50/2 = 25 body-lengths.
    For a weta on 800 mm: ratio = 800 / body_mm.
    Simulated gradient = DROSO_BODY * ratio = 2 * (800 / body_mm).
    """
    ratio = real_gradient_mm / species_body_mm
    return DROSO_BODY_MM * ratio


def compute_scaled_conductance_params(species_body_mm):
    """Scale the Drosophila cylinder model to the weta body size.

    We scale isotropically: if the weta is k× longer than Drosophila,
    radius and length both scale by k.

    Returns (surface_m2, thickness_m, j_to_degC)
        surface_m2 : surface area of scaled cylinder [m²]
        thickness_m : wall-to-wall thickness (= diameter) [m]
        j_to_degC  : °C temperature change per Joule of heat added
    """
    k = species_body_mm / DROSO_BODY_MM

    # Scaled cylinder dimensions
    r = DROSO_RADIUS_MM * k * 1e-3   # [m]
    l = DROSO_BODY_MM * k * 1e-3     # [m]  (= species_body_mm * 1e-3)

    surface = 2 * np.pi * r * l + 2 * np.pi * r**2   # full cylinder surface [m²]
    thickness = 2 * r                                  # diameter [m]

    # Mass of water cylinder [kg]
    volume_m3 = np.pi * r**2 * l
    mass_kg = volume_m3 * 1000.0  # density of water = 1000 kg/m³

    # 1 J heats mass_kg of water by 1/(mass_kg * 4186) °C
    # (specific heat of water = 4186 J/(kg·°C))
    j_to_degC = 1.0 / (mass_kg * 4186.0)

    return surface, thickness, j_to_degC


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
# Vectorized heat conduction — NOW WITH SPECIES-SCALED PARAMETERS
# ──────────────────────────────────────────────────────────────────────────────

def _update_body_temp(xp, body_t, ambient_t, duration,
                      surface_m2, thickness_m, j_to_degC):
    """Vectorized body temperature update using scaled conductance model.

    Heat flow equation (Giraldo et al. Eq. 4):
        Q = lambda * A * (T1 - T2) / D * t

    where lambda = 0.6 W/(m·K), A = surface, D = thickness, t = duration.
    Temperature change = Q * j_to_degC.
    """
    dT = ambient_t - body_t
    Q = WATER_CONDUCTANCE * surface_m2 * (dT / thickness_m) * duration
    temp_change = Q * j_to_degC

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
# Vectorized simulation — NOW WITH SPECIES-SPECIFIC SCALING
# ──────────────────────────────────────────────────────────────────────────────

def simulate_null_vectorized(n_sim, gradient_ext, gradient_dist,
                              walk_dur, rearing_t, sps,
                              surface_m2, thickness_m, j_to_degC,
                              use_gpu=False):
    """Simulate n_sim flies with no temperature preference in a gradient.

    Parameters
    ----------
    gradient_dist : float
        The SCALED gradient length in mm (computed from body-size ratio).
    surface_m2, thickness_m, j_to_degC : float
        Species-scaled conductance parameters.

    Returns
    -------
    null_mean, null_sem, bins
    """
    from scipy import stats as sp_stats

    xp, backend = get_backend(use_gpu)

    v_params, d_params = _calc_parameters(rearing_t)
    deg_per_mm = abs(gradient_ext[1] - gradient_ext[0]) / gradient_dist

    # ── Initial state for all flies ──
    position = xp.random.uniform(0, gradient_dist, size=n_sim)
    ambient_t = position * deg_per_mm + gradient_ext[0]
    body_t = ambient_t.copy()
    time_acc = xp.zeros(n_sim)
    alive = xp.ones(n_sim, dtype=bool)

    # ── Histogram accumulators ──
    bin_num = int(gradient_ext[1] - gradient_ext[0])
    bins_np = np.linspace(gradient_ext[0] - 0.5, gradient_ext[1] + 0.5, bin_num + 2)
    hist_matrix = np.zeros((n_sim, bin_num + 1))

    steps_done = 0
    pbar = tqdm(total=100, desc="IGLOO simulation", unit="%",
                bar_format="{l_bar}{bar}| {n:.0f}/{total}% [{elapsed}<{remaining}]")

    while True:
        n_alive = int(xp.sum(alive))
        if n_alive == 0:
            break

        pct = int(100 * (1 - n_alive / n_sim))
        if pct > pbar.n:
            pbar.update(pct - pbar.n)

        # ── Random numbers ──
        p_vel = xp.random.random(n_sim)
        p_dur = xp.random.random(n_sim)
        direction = xp.where(xp.random.random(n_sim) < 0.5, -1.0, 1.0)

        # ── Velocity and duration from body temperature ──
        velocity = _vel_func(xp, p_vel, body_t, v_params)
        duration = _dur_func(xp, p_dur, body_t, d_params)
        velocity = xp.maximum(velocity, 0.0)
        duration = xp.maximum(duration, 0.001)

        # ── Update time ──
        time_acc += duration * alive
        newly_done = alive & (time_acc >= walk_dur)

        # ── Update position with reflective boundaries ──
        step = direction * velocity * duration * alive
        new_pos = position + step

        overshoot_far = new_pos > gradient_dist
        new_pos = xp.where(overshoot_far, 2 * gradient_dist - new_pos, new_pos)
        overshoot_near = new_pos < 0
        new_pos = xp.where(overshoot_near, -new_pos, new_pos)
        new_pos = xp.clip(new_pos, 0, gradient_dist)

        position = new_pos

        # ── Update ambient temperature from position ──
        ambient_t = position * deg_per_mm + gradient_ext[0]

        # ── Update body temperature — SCALED conductance ──
        body_t = _update_body_temp(xp, body_t, ambient_t, duration,
                                    surface_m2, thickness_m, j_to_degC)

        # ── Accumulate into histograms ──
        if backend == "gpu":
            amb_cpu = xp.asnumpy(ambient_t)
            dur_cpu = xp.asnumpy(duration)
            alive_cpu = xp.asnumpy(alive)
        else:
            amb_cpu = ambient_t
            dur_cpu = duration
            alive_cpu = alive

        bin_indices = np.searchsorted(bins_np, amb_cpu) - 1
        bin_indices = np.clip(bin_indices, 0, bin_num)
        fly_indices = np.where(alive_cpu)[0]
        np.add.at(hist_matrix, (fly_indices, bin_indices[fly_indices]),
                  dur_cpu[fly_indices])

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
    """Per-animal normalised histograms → mean ± SEM."""
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

def correct_and_plot(species_temps, species_nulls, bins, output_dir):
    """For each species: subtract its own null → preference index, plot, save CSV.

    species_nulls : dict  {species: (null_mean, null_sem)}
    """
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    csv_rows = []

    for sp in SPECIES_ORDER:
        if sp not in species_temps or len(species_temps[sp]) == 0:
            continue
        if sp not in species_nulls:
            print(f"  [WARN] No null model for {sp}, skipping")
            continue

        null_mean, null_sem = species_nulls[sp]
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

        body_mm = SPECIES_BODY_LENGTH_MM.get(sp, 50.0)
        scaled_grad = compute_scaled_gradient(body_mm)

        print(f"\n  {sp} (n={n_animals}, body={body_mm:.0f} mm, "
              f"scaled gradient={scaled_grad:.1f} mm):")
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
                 label=f"IGLOO null (grad={scaled_grad:.0f} mm)")
        if not np.isnan(tp_corrected):
            ax1.axvline(tp_corrected, color="darkred", ls="--", lw=1,
                        label=f"$T_p$ corrected = {tp_corrected:.1f} °C")
        ax1.set_ylabel("Probability density")
        ax1.set_title(f"Temperature preference — {sp} "
                      f"(body {body_mm:.0f} mm, gradient scaled to {scaled_grad:.0f} mm)")
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
        description="Step 4: IGLOO cold-trapping correction for weta "
                    "(body-size-scaled null model)."
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
    parser.add_argument("--real_gradient_mm", type=float, default=555.0,
                        help="Real experimental gradient length in mm (default: 555)")

    args = parser.parse_args()

    # Override module-level constant if user provides a different value
    global REAL_GRADIENT_MM
    REAL_GRADIENT_MM = args.real_gradient_mm

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

    # ── 2. Per-species IGLOO null model ──
    species_nulls = {}

    for sp in SPECIES_ORDER:
        if sp not in species_temps or len(species_temps[sp]) == 0:
            continue

        body_mm = SPECIES_BODY_LENGTH_MM.get(sp, 50.0)
        scaled_grad = compute_scaled_gradient(body_mm)
        surface, thickness, j2degc = compute_scaled_conductance_params(body_mm)

        print(f"\n── Running IGLOO null for {sp} ──")
        print(f"  Body length:       {body_mm:.0f} mm")
        print(f"  Real gradient:     {REAL_GRADIENT_MM:.0f} mm")
        print(f"  Body-length ratio: {REAL_GRADIENT_MM / body_mm:.1f}×")
        print(f"  Scaled gradient:   {scaled_grad:.1f} mm "
              f"(= {DROSO_BODY_MM} mm × {REAL_GRADIENT_MM / body_mm:.1f})")
        print(f"  Conductance: surface={surface*1e6:.2f} mm², "
              f"thickness={thickness*1e3:.2f} mm, "
              f"J→°C={j2degc:.4f}")

        null_mean, null_sem, bins = simulate_null_vectorized(
            n_sim=args.n_sim,
            gradient_ext=(t_min_round, t_max_round),
            gradient_dist=scaled_grad,
            walk_dur=args.walk_dur,
            rearing_t=args.rearing_t,
            sps=args.sps,
            surface_m2=surface,
            thickness_m=thickness,
            j_to_degC=j2degc,
            use_gpu=args.use_gpu,
        )
        species_nulls[sp] = (null_mean, null_sem)
        print(f"  Null model: {len(null_mean)} bins")

    # ── 3. Correct and plot (per-species null) ──
    print("\n── Correcting for cold-trapping ──")
    correct_and_plot(species_temps, species_nulls, bins, output_dir)

    # ── 4. Save scaling parameters for reference ──
    scaling_info = {}
    for sp in SPECIES_ORDER:
        body_mm = SPECIES_BODY_LENGTH_MM.get(sp, 50.0)
        surface, thickness, j2degc = compute_scaled_conductance_params(body_mm)
        scaling_info[sp] = {
            "body_length_mm": body_mm,
            "real_gradient_mm": REAL_GRADIENT_MM,
            "body_length_ratio": REAL_GRADIENT_MM / body_mm,
            "scaled_gradient_mm": compute_scaled_gradient(body_mm),
            "droso_ref_gradient_mm": 50.0,
            "droso_ref_body_mm": DROSO_BODY_MM,
            "conductance_surface_m2": surface,
            "conductance_thickness_m": thickness,
            "conductance_j_to_degC": j2degc,
        }

    json_path = os.path.join(output_dir, "igloo_scaling_parameters.json")
    with open(json_path, "w") as f:
        json.dump(scaling_info, f, indent=2)
    print(f"\n  Scaling parameters → {json_path}")

    print(f"\n[DONE] All outputs in {output_dir}/")


if __name__ == "__main__":
    main()

#Example command to run: python weta_igloo_correction.py     --base_dir /home/geuba03p/weta_project/weta_videos_cropped     --n_sim 10000 --walk_dur 3600
#Example env: conda activate deer_project_2