# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Created on Tue Feb  3 14:31:12 2026

@author: km923
"""

# -*- coding: utf-8 -*-
"""

two_agent_mine_plant_rl.py

Two-agent RL scheduling engine:
  - Plant agent chooses a yearly plant mode (target grade + reclaim preference + penalty weights).
  - Mining agent chooses min-width compliant panels/clusters under:
        precedence (always),
        annual mining capacity (hard),
        minimum mining width (hard if Condition==1),
        max allowable depth / exposed height (hard if indicator==1),
        optional contiguity (hard if Condition==1).

Key design decisions:
  1) Actions are CLUSTERS (panels) not single blocks (prevents isolated "1 block/year").
  2) Environment auto-fills remaining capacity greedily after each mining action.
  3) Reward is plant-level discounted cashflow from processed material after blending
     mined ore + reclaimed stockpiles to meet plant grade window.

Dependencies: numpy, pandas, scipy (optional), torch
"""



import os
import ast
import math
import time
import json
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set

import numpy as np
import pandas as pd
from collections import OrderedDict

# Optional torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ============================================================
# 0) TWEAKING KNOBS (start here)
# ============================================================

TWEAK = dict(
    # ---------- RL ----------
    epochs=5,
    lr=3e-4,
    rl_gamma=0.99,
    gae_lambda=0.95,
    entropy_coef=0.01,
    max_steps_per_episode=1000,

    # ---------- Mining auto-fill (prevents "1 block/year") ----------
    autofill_enabled=True,
    autofill_max_additions=2000,     # max greedy add-ons per step
    autofill_min_remaining_frac=0.01,  # stop autofill if remaining capacity < 3%
    autofill_score_w_value=1.0,
    autofill_score_w_grade=0.7,
    autofill_score_w_prox=0.3,

    # ---------- Minimum mining width enforcement ----------
    # If Condition==1, you want "panel mining" and no thin components at year-end.
    # Cluster builder enforces bbox >= min_width in BOTH X and Y.
    cluster_fill_ratio_min=0.60,      # merge clusters with poor fill ratio (finger-like)
    cluster_merge_passes=2,

    # ---------- Contiguity / faces ----------
    # Hard mode (Condition=1): require same-year contiguity (like a face),
    # but allow a few new face seeds to avoid deadlocks.
    allow_multi_faces=True,
    max_face_seeds_per_year=10,        # allows mining disjoint faces in same year (still panel-based)

    # ---------- Deadlock control ----------
    # If no feasible clusters fit remaining capacity/height, commit year and roll.
    max_years_cap=5000,               # safety cap on horizon

    # ---------- Reward shaping ----------
    # Penalties for under-using annual mining capacity and plant capacity:
    penalty_mining_underuse=12.0,
    penalty_plant_underuse=3.0,
    allow_lonely_seed_on_deadlock=True,   # allow a seed even if no neighbour can fit
    penalty_lonely_seed=5000.0,           # discourage unless necessary (tune)
    autofill_score_w_fit=2.0,
    clusters_per_year_target=12.0,   # more, smaller clusters => better packing
    cluster_min_t_frac=0.15,         # merge less aggressively on tonnage

    # ---------- Plant modes ----------
    # Discrete modes for plant agent:
    # - target grade location within [low, up]
    # - reclaim bias (prefer reclaim vs prefer mined)
    # Map of mode -> (target_grade_frac, reclaim_bias)
    # target_grade_frac: 0=low, 0.5=mid, 1=up
    plant_modes=[
        (0.50, 0.0),  # mid, neutral
        (0.25, 0.0),  # closer low, neutral
        (0.75, 0.0),  # closer high, neutral
        (0.50, 1.0),  # mid, reclaim-heavy
        (0.25, 1.0),  # low, reclaim-heavy
        (0.75, 1.0),  # high, reclaim-heavy
        (0.50, -1.0), # mid, mine-heavy
        (0.25, -1.0), # low, mine-heavy
        (0.75, -1.0), # high, mine-heavy
    ],
)

# --- Belief / uncertainty ---
TWEAK.update(dict(
    use_belief_state=True,
    n_scenarios=20,                 # training ensemble size
    n_eval_scenarios=50,            # evaluation ensemble size (later)
    belief_features_local=True,     # per-cluster features into mining_obs
    belief_features_global=True,    # global features into plant_obs
    observation_noise_grade=0.05,   # relative noise for "sensor" grade
    assimilation_interval="year",   # "year" is simplest initially
    belief_prior_sigma_rel=0.10,    # prior scenario spread around initial grade
))

# ============================================================
# Logging
# ============================================================

def mk_logger(name="two_agent_mineplan", level=logging.INFO):
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(level)
    lg.propagate = False
    return lg

LOG = mk_logger()


# ============================================================
# 1) Configuration
# ============================================================

class Configuration:
    """Simple key=value config reader with Python-literal parsing."""
    input_file_name_path: str = "BH_realizations_3D.csv"
    output_file_name_path: str = "final_results.xlsx"
    split_char: str = ","

    # economics
    price: float = 0.0
    refining_cost: float = 0.0
    recovery: float = 0.0
    p_cost: float = 0.0
    mining_cost: float = 0.0
    rehandling_cost: float = 0.0
    dis_rate: float = 0.0

    # physical / mining
    mineral_density: float = 0.0
    waste_density: float = 0.0
    cutoff: float = 0.0
    mining_capacity: float = 0.0
    process_capacity: float = 0.0

    minimum_mining_width_defined: float = 0.0
    Condition: int = 0

    max_allowable_depth: float = 0.0                    # in "number of vertical blocks"
    max_allowable_depth_priority_indicator: int = 0

    user_define_low_limit: float = 0.0
    user_define_up_limit: float = 0.0

    # stockpiles
    numb_of_stockpiles: int = 0
    stockpiles: List[Tuple[float, float]] = ()
    stockpiles_capacity: List[float] = ()
    # slope precedence (optional)
    use_slope_precedence: int = 1            # 1=build/merge slope-based precedence
    slopes: List[float] = None               # [N, E, S, W] in degrees (from horizontal)
    slope_z_window_blocks: int = 2          # how many vertical block layers above to search
    precedence_same_year_ok: int = 1   # 1: pred may be same year; 0: pred must be earlier year
    capacity_rescue_enabled=True
    capacity_rescue_min_unused_frac=0.05   # only rescue if >5% cap still unused
    capacity_rescue_max_additions=2000     # safety
    capacity_rescue_penalty=500.0          # per lonely-seed used in rescue (soft discouragement)

    def __init__(self, config_file_path: str):
        if not os.path.isfile(config_file_path):
            raise FileNotFoundError(f"Config not found: {config_file_path}")

        # NEW: remember config path and ALL raw/parsed values from file (even unknown keys)
        self.config_file_path = config_file_path
        self.cfg_raw = OrderedDict()     # key -> raw string (after stripping comments)
        self.cfg_parsed = OrderedDict()  # key -> parsed python value (best-effort)

        with open(config_file_path, encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                k, v = [t.strip() for t in line.split("=", 1)]
                v = v.split("#", 1)[0].strip()

                # store raw and parsed values (ALL keys)
                self.cfg_raw[k] = v
                try:
                    vv = ast.literal_eval(v)
                except Exception:
                    vv = v
                self.cfg_parsed[k] = vv

                # existing behaviour: only assign if class has the attribute
                if k == "blender_mode":
                    setattr(self, k, str(vv).strip().lower())
                    continue

                if hasattr(self, k):
                    setattr(self, k, vv)


# ============================================================
# 2) Data parsing helpers
# ============================================================

def parse_listlike(x) -> List[int]:
    if isinstance(x, (list, tuple, set, np.ndarray, pd.Series)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []
    try:
        v = ast.literal_eval(s)
        return parse_listlike(v)
    except Exception:
        pass
    s = s.strip("[]{}()")
    parts = [p for p in re_split(r"[,\s;]+", s) if p]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            try:
                out.append(int(float(p)))
            except Exception:
                pass
    return out

def re_split(pattern: str, text: str) -> List[str]:
    import re
    return re.split(pattern, text)

def _allowable_slope_cardinal(dx: float, dy: float, slopes_nesw: List[float]) -> float:
    """
    slopes_nesw = [N, E, S, W] in degrees (from horizontal).
    Uses +Y as North, +X as East. Interpolates linearly between cardinals by bearing.
    """
    sN, sE, sS, sW = [float(s) for s in slopes_nesw]

    # Bearing: 0=N, 90=E, 180=S, 270=W (clockwise)
    b = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0

    if b < 90.0:
        t = b / 90.0
        return sN + t * (sE - sN)
    elif b < 180.0:
        t = (b - 90.0) / 90.0
        return sE + t * (sS - sE)
    elif b < 270.0:
        t = (b - 180.0) / 90.0
        return sS + t * (sW - sS)
    else:
        t = (b - 270.0) / 90.0
        return sW + t * (sN - sW)


def build_slope_precedence_blocks(
    df: pd.DataFrame,
    slopes_nesw: List[float],
    z_window_blocks: int,
    dx: float,
    dy: float,
    dz: float,
) -> Dict[int, List[int]]:
    """
    Build block-level slope precedence:
      For a lower block i, any upper block j (higher Z) that lies within the
      slope cone must be mined before i.

    Condition:
      act = atan2(dz_ij, dxy_ij)  (angle from horizontal)
      if act >= allowable(bearing) => j is a predecessor of i

    Uses KDTree per Z-slice (scipy if available). If scipy isn't available,
    falls back to a slower brute approach within the z-window.
    """
    slopes_nesw = [float(s) for s in slopes_nesw]
    min_ang = max(1e-3, min(slopes_nesw))  # guard against 0
    tan_min = math.tan(math.radians(min_ang))

    # Pad radius slightly to avoid missing boundary cases
    pad = 0.5 * math.hypot(dx, dy)

    # Prepare per-level XY structures
    levels = sorted(df["Z"].unique().tolist())  # ascending: low -> high
    by_z = {z: sub for z, sub in df.groupby("Z", sort=False)}

    # Try KDTree
    try:
        from scipy.spatial import cKDTree  # type: ignore
        has_kdtree = True
    except Exception:
        has_kdtree = False

    level_data = []
    for z in levels:
        sub = by_z[z]
        ids = sub["Indices"].astype(int).to_numpy()
        xy = sub[["X", "Y"]].to_numpy(dtype=float)
        tree = None
        if has_kdtree and len(xy) > 0:
            tree = cKDTree(xy)
        level_data.append((float(z), ids, xy, tree))

    # Map: block id -> (x,y,z)
    # (fast lookup arrays)
    idx = df.set_index("Indices")
    x_map = idx["X"].to_dict()
    y_map = idx["Y"].to_dict()
    z_map = idx["Z"].to_dict()

    preds: Dict[int, List[int]] = {int(b): [] for b in df["Indices"].astype(int).tolist()}

    # For each lower level k, look at upper levels k+1..k+z_window
    L = len(level_data)
    for k in range(L):
        z0, ids0, xy0, _ = level_data[k]
        # upper levels have larger Z
        hi = min(L, k + 1 + int(max(1, z_window_blocks)))
        for u in range(k + 1, hi):
            zu, idsu, xyu, treeu = level_data[u]
            dv = float(zu - z0)
            if dv <= 1e-12:
                continue

            # Max radius using flattest (min) slope angle
            r_max = dv / tan_min + pad

            if len(idsu) == 0:
                continue

            if has_kdtree and treeu is not None:
                # query candidates for each point in this lower slice
                for i_local, bid in enumerate(ids0):
                    x0, y0 = float(xy0[i_local, 0]), float(xy0[i_local, 1])
                    cand_idx = treeu.query_ball_point([x0, y0], r=r_max)
                    if not cand_idx:
                        continue
                    for j_local in cand_idx:
                        up_id = int(idsu[j_local])
                        dx_ = float(x_map[up_id] - x0)
                        dy_ = float(y_map[up_id] - y0)
                        dxy = math.hypot(dx_, dy_)
                        if dxy <= 1e-12:
                            act = 90.0
                        else:
                            act = math.degrees(math.atan2(dv, dxy))
                        allo = _allowable_slope_cardinal(dx_, dy_, slopes_nesw)
                        if act >= allo - 1e-12:
                            preds[int(bid)].append(up_id)
            else:
                # slower brute fallback within r_max
                for bid in ids0:
                    x0, y0 = float(x_map[int(bid)]), float(y_map[int(bid)])
                    for up_id in idsu:
                        up_id = int(up_id)
                        dx_ = float(x_map[up_id] - x0)
                        dy_ = float(y_map[up_id] - y0)
                        dxy = math.hypot(dx_, dy_)
                        if dxy > r_max + 1e-12:
                            continue
                        act = 90.0 if dxy <= 1e-12 else math.degrees(math.atan2(dv, dxy))
                        allo = _allowable_slope_cardinal(dx_, dy_, slopes_nesw)
                        if act >= allo - 1e-12:
                            preds[int(bid)].append(up_id)

    # de-dup + sort
    for b in preds:
        if preds[b]:
            preds[b] = sorted(set(int(x) for x in preds[b]))
    return preds


def load_blocks_csv(cfg: Configuration) -> pd.DataFrame:
    path = cfg.input_file_name_path
    if not path:
        raise ValueError("input_file_name_path is empty in config.")
    df = pd.read_csv(path, sep=(cfg.split_char or ","), engine="python", encoding="unicode_escape")
    return df

def preprocess_blocks(df_raw: pd.DataFrame, cfg: Configuration) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, List[int]], Dict[str, float]]:
    """
    Required columns:
      Indices, X, Y, Z, Grade, x_axis_len, y_axis_len, z_axis_len, precedence index, adjacent index
    """
    df = df_raw.copy()

    req = {"Indices","X","Y","Z","Grade","x_axis_len","y_axis_len","z_axis_len"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    # list columns optional but strongly recommended
    if "precedence index" not in df.columns:
        df["precedence index"] = [[] for _ in range(len(df))]
    if "adjacent index" not in df.columns:
        df["adjacent index"] = [[] for _ in range(len(df))]

    # types
    df["Indices"] = pd.to_numeric(df["Indices"], errors="coerce").astype(int)
    for c in ["X","Y","Z","Grade","x_axis_len","y_axis_len","z_axis_len"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # parse lists
    df["precedence index"] = df["precedence index"].apply(parse_listlike)
    df["adjacent index"] = df["adjacent index"].apply(parse_listlike)

    # compute tonnage & value
    vol = df["x_axis_len"] * df["y_axis_len"] * df["z_axis_len"]
    ore = df["Grade"] >= float(cfg.cutoff)

    df["Tonnage"] = np.where(
        ore,
        vol * float(cfg.mineral_density),
        vol * float(cfg.waste_density),
    )

    ore_margin_per_t = (
        (float(cfg.price) - float(cfg.refining_cost)) * df["Grade"] * (float(cfg.recovery) / 100.0)
        - float(cfg.mining_cost) - float(cfg.p_cost)
    )
    waste_margin_per_t = -float(cfg.mining_cost)
    df["Block value"] = np.where(ore, ore_margin_per_t * df["Tonnage"], waste_margin_per_t * df["Tonnage"])

    # sort: top-down
    df = df.sort_values(["Z","X","Y"], ascending=[False, True, True]).reset_index(drop=True)
    dx = float(df["x_axis_len"].iloc[0])
    dy = float(df["y_axis_len"].iloc[0])
    dz = float(df["z_axis_len"].iloc[0])
    # ---------- OPTIONAL: build/merge slope-based precedence ----------
    if int(getattr(cfg, "use_slope_precedence", 0)) == 1:
        slopes = getattr(cfg, "slopes", None)
        if slopes is None:
            slopes = cfg.cfg_parsed.get("slopes", None)
    
        if slopes is not None and isinstance(slopes, (list, tuple)) and len(slopes) == 4:
            zwin = int(getattr(cfg, "slope_z_window_blocks", 12) or 12)
            LOG.info(f"Building slope precedence from slopes=[N,E,S,W]={list(slopes)} with z_window_blocks={zwin} ...")
    
            slope_preds = build_slope_precedence_blocks(df, list(slopes), zwin, dx, dy, dz)
    
            # merge with any existing precedence index
            merged = []
            for _, r in df.iterrows():
                b = int(r["Indices"])
                old = list(r["precedence index"]) if isinstance(r["precedence index"], list) else []
                add = slope_preds.get(b, [])
                merged.append(sorted(set(int(x) for x in (old + add))))
            df["precedence index"] = merged
        else:
            LOG.warning("use_slope_precedence=1 but slopes is missing/invalid; expected slopes=[N,E,S,W].")


    precedence = {int(r.Indices): [int(x) for x in r["precedence index"]] for _, r in df.iterrows()}
    adjacency  = {int(r.Indices): [int(x) for x in r["adjacent index"]]   for _, r in df.iterrows()}

    

    min_w = float(cfg.minimum_mining_width_defined or max(dx, dy))
    # Convert max_allowable_depth in "number of blocks" -> meters
    H_max = None
    if int(cfg.max_allowable_depth_priority_indicator) == 1 and float(cfg.max_allowable_depth) > 0 and dz > 0:
        H_max = float(cfg.max_allowable_depth) * dz

    const = dict(dx=dx, dy=dy, dz=dz, min_w=min_w, H_max=(H_max if H_max is not None else -1.0))
    return df, precedence, adjacency, const


# ============================================================
# 3) Clustering into min-width panels (actions)
# ============================================================


from dataclasses import field

@dataclass
class BeliefState:
    # required
    cluster_grade_scen: np.ndarray  # [C,S]

    # optional (extend later)
    cluster_recovery_scen: Optional[np.ndarray] = None
    cluster_proc_factor_scen: Optional[np.ndarray] = None

    # cached summaries
    mu_grade: np.ndarray = field(default_factory=lambda: np.array([]))
    sd_grade: np.ndarray = field(default_factory=lambda: np.array([]))
    q10_grade: np.ndarray = field(default_factory=lambda: np.array([]))
    q90_grade: np.ndarray = field(default_factory=lambda: np.array([]))

    def recompute_summaries(self):
        g = self.cluster_grade_scen
        self.mu_grade = g.mean(axis=1)
        self.sd_grade = g.std(axis=1)
        self.q10_grade = np.quantile(g, 0.10, axis=1)
        self.q90_grade = np.quantile(g, 0.90, axis=1)


# --- Replace your Cluster class with this ---
@dataclass
class Cluster:
    cid: int
    z: float
    blocks: list[int]
    tonnage: float
    value: float
    ore_tonnage: float
    ore_grade: float
    cx: float
    cy: float
    fill_ratio: float
    bbox_wx: float
    bbox_wy: float
    avg_grade: float = 0.0

def build_min_width_clusters(
    blocks_df: pd.DataFrame,
    cfg: Configuration,
    const: Dict[str, float],
) -> Tuple[List[Cluster], Dict[int, int]]:
    """
    Tile-based clustering (tight panels):
      - Build clusters as min-width tiles: tile = minimum_mining_width_defined (e.g., 20m)
      - Each elevation (Z) is clustered independently
      - Grade-aware merge of undersized / sparse tiles to avoid isolated leftovers
      - Result: many small, packable actions so yearly mining can reach mining_capacity (~40k) much better.

    Returns:
      clusters: list[Cluster]
      block_to_cluster: dict block_id -> cluster_id
    """
    dx, dy = float(const["dx"]), float(const["dy"])
    min_w = float(const["min_w"])

    # --- tile size: tight clusters 20x20 (or whatever min_w is) ---
    tile = max(min_w, dx, dy)

    # --- packability targets (only used for merge decisions) ---
    cap = float(cfg.mining_capacity)
    # Aim for ~6 clusters/year by default => ~6.7k each for a 40k cap
    clusters_per_year_target = float(TWEAK.get("clusters_per_year_target", 12.0))
    target_t = cap / max(1.0, clusters_per_year_target)
    min_t = float(TWEAK.get("cluster_min_t_frac", 0.15)) * target_t
    # (upper bound split not needed because tiles are already tight)

    cutoff = float(cfg.cutoff)

    df = blocks_df[["Indices","X","Y","Z","Grade","Tonnage","Block value"]].copy()
    df["Indices"] = df["Indices"].astype(int)

    # Align tiles to a fixed origin (global) so tile boundaries are consistent
    x0 = float(df["X"].min())
    y0 = float(df["Y"].min())

    # expected cells count in one tile (for fill ratio proxy)
    nx_tile = max(1, int(round(tile / dx)))
    ny_tile = max(1, int(round(tile / dy)))
    expected_cells = float(nx_tile * ny_tile)

    # temporary structure: per-z map from (gx,gy) -> cluster object with extra fields
    clusters_tmp = []
    # We'll store cell keys for merging
    # Each element: (z, gx, gy, blocks_list, ...)

    def make_cluster(zv: float, gx: int, gy: int, bd: pd.DataFrame) -> dict:
        ids = bd["Indices"].astype(int).tolist()
        xs = bd["X"].to_numpy(dtype=float)
        ys = bd["Y"].to_numpy(dtype=float)

        bbox_wx = float(xs.max() - xs.min() + dx) if len(xs) else float(tile)
        bbox_wy = float(ys.max() - ys.min() + dy) if len(ys) else float(tile)

        ton = float(bd["Tonnage"].sum())
        val = float(bd["Block value"].sum())

        ore_bd = bd[bd["Grade"] >= cutoff]
        ore_t = float(ore_bd["Tonnage"].sum())
        ore_g = 0.0 if ore_t <= 1e-12 else float((ore_bd["Tonnage"] * ore_bd["Grade"]).sum() / ore_t)

        fill_ratio = float(len(ids) / max(1.0, expected_cells))

        return dict(
            z=float(zv), gx=int(gx), gy=int(gy),
            blocks=[int(b) for b in ids],
            tonnage=ton, value=val,
            ore_tonnage=ore_t, ore_grade=ore_g,
            cx=float(xs.mean()) if len(xs) else float(x0 + (gx + 0.5)*tile),
            cy=float(ys.mean()) if len(ys) else float(y0 + (gy + 0.5)*tile),
            fill_ratio=fill_ratio,
            bbox_wx=bbox_wx, bbox_wy=bbox_wy,
        )

    # --- 1) initial tile clusters per elevation ---
    for z, sub in df.groupby("Z", sort=False):
        sub = sub.copy()
        sub["gx"] = np.floor((sub["X"] - x0) / tile).astype(int)
        sub["gy"] = np.floor((sub["Y"] - y0) / tile).astype(int)

        for (gx, gy), bd in sub.groupby(["gx","gy"], sort=False):
            if len(bd) == 0:
                continue
            clusters_tmp.append(make_cluster(float(z), int(gx), int(gy), bd))

    # Helper for recomputing a merged cluster metrics
    df_idx = df.set_index("Indices")

    def recompute(c: dict):
        bd = df_idx.loc[c["blocks"]]
        xs = bd["X"].to_numpy(dtype=float)
        ys = bd["Y"].to_numpy(dtype=float)
        c["cx"] = float(xs.mean())
        c["cy"] = float(ys.mean())
        c["bbox_wx"] = float(xs.max() - xs.min() + dx)
        c["bbox_wy"] = float(ys.max() - ys.min() + dy)
        c["fill_ratio"] = float(len(c["blocks"]) / max(1.0, expected_cells))
        c["tonnage"] = float(bd["Tonnage"].sum())
        c["value"] = float(bd["Block value"].sum())
        ore_bd = bd[bd["Grade"] >= cutoff]
        c["ore_tonnage"] = float(ore_bd["Tonnage"].sum())
        if c["ore_tonnage"] <= 1e-12:
            c["ore_grade"] = 0.0
        else:
            c["ore_grade"] = float((ore_bd["Tonnage"] * ore_bd["Grade"]).sum() / c["ore_tonnage"])

    # --- 2) grade-aware merge of undersized/sparse tiles (same Z only) ---
    # Reason: avoid isolated leftover tiles & make clusters less “needle-like”.
    # Neighbor candidates are the 8 surrounding tiles.
    def merge_pass():
        nonlocal clusters_tmp
        # index by (z,gx,gy)
        idx = {(c["z"], c["gx"], c["gy"]): k for k, c in enumerate(clusters_tmp)}
        alive = np.ones(len(clusters_tmp), dtype=bool)

        # iterate small first
        order = np.argsort([c["tonnage"] for c in clusters_tmp])
        merged_any = False

        for k in order:
            if not alive[k]:
                continue
            c = clusters_tmp[k]
            # merge if too small OR too sparse (fill ratio low)
            if not (c["tonnage"] < min_t or c["fill_ratio"] < float(TWEAK.get("cluster_fill_ratio_min", 0.60))):
                continue

            z = c["z"]
            gx = c["gx"]
            gy = c["gy"]

            # find best neighbor among 8-neighborhood
            best_j = None
            best_score = 1e18
            for dxg in (-1, 0, 1):
                for dyg in (-1, 0, 1):
                    if dxg == 0 and dyg == 0:
                        continue
                    key = (z, gx + dxg, gy + dyg)
                    j = idx.get(key, None)
                    if j is None or (not alive[j]):
                        continue
                    n = clusters_tmp[j]

                    # grade-aware + proximity (same Z so distance is only XY)
                    dg = abs(float(c["ore_grade"]) - float(n["ore_grade"]))
                    dxy = math.hypot(float(c["cx"]) - float(n["cx"]), float(c["cy"]) - float(n["cy"]))
                    # weight grade similarity strongly; distance breaks ties
                    score = 5.0 * dg + 0.001 * dxy
                    if score < best_score:
                        best_score = score
                        best_j = j

            if best_j is None:
                continue

            # merge k -> best_j
            clusters_tmp[best_j]["blocks"].extend(c["blocks"])
            alive[k] = False
            merged_any = True
            recompute(clusters_tmp[best_j])

        # keep only alive
        if merged_any:
            clusters_tmp = [c for i, c in enumerate(clusters_tmp) if alive[i]]
        return merged_any

    # number of merge passes (use your existing knob)
    for _ in range(int(TWEAK.get("cluster_merge_passes", 2))):
        if not merge_pass():
            break

    # --- 3) finalize into Cluster objects with sequential cids ---
    clusters: List[Cluster] = []
    block_to_cluster: Dict[int, int] = {}
    clusters_tmp = sorted(clusters_tmp, key=lambda c: (c["z"], c["gx"], c["gy"]))

    for cid, c in enumerate(clusters_tmp):
        cl = Cluster(
            cid=int(cid),
            z=float(c["z"]),
            blocks=[int(b) for b in c["blocks"]],
            tonnage=float(c["tonnage"]),
            value=float(c["value"]),
            ore_tonnage=float(c["ore_tonnage"]),
            ore_grade=float(c["ore_grade"]),
            cx=float(c["cx"]),
            cy=float(c["cy"]),
            fill_ratio=float(c["fill_ratio"]),
            bbox_wx=float(c["bbox_wx"]),
            bbox_wy=float(c["bbox_wy"]),
        )
        clusters.append(cl)
        for b in cl.blocks:
            block_to_cluster[int(b)] = int(cid)

    LOG.info(f"Built {len(clusters)} clusters (tile={tile:.2f}m, target_cluster_t≈{target_t:.0f}t).")
    # Optional: print cluster tonnage stats
    tons = np.array([c.tonnage for c in clusters], float)
    LOG.info(f"Cluster tonnage stats: min={tons.min():.1f}, p50={np.median(tons):.1f}, p90={np.percentile(tons,90):.1f}, max={tons.max():.1f}")
    return clusters, block_to_cluster



# ============================================================
# 4) Cluster graph: precedence + adjacency at cluster level
# ============================================================

def build_cluster_graph(
    clusters: List[Cluster],
    block_to_cluster: Dict[int, int],
    precedence_blocks: Dict[int, List[int]],
    adjacency_blocks: Dict[int, List[int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    cluster_predecessors[c] = list of clusters that must be mined before c
    cluster_neighbors[c]    = neighbor clusters (for contiguity/face)
    """
    # cluster predecessors
    cluster_preds: Dict[int, Set[int]] = {c.cid: set() for c in clusters}
    for b, preds in precedence_blocks.items():
        cb = block_to_cluster.get(int(b))
        if cb is None:
            continue
        for p in preds:
            cp = block_to_cluster.get(int(p))
            if cp is None:
                continue
            if cp != cb:
                cluster_preds[cb].add(cp)

    cluster_preds_list = {k: sorted(v) for k, v in cluster_preds.items()}

    # cluster neighbors
    cluster_nbrs: Dict[int, Set[int]] = {c.cid: set() for c in clusters}
    for b, nbrs in adjacency_blocks.items():
        cb = block_to_cluster.get(int(b))
        if cb is None:
            continue
        for n in nbrs:
            cn = block_to_cluster.get(int(n))
            if cn is None or cn == cb:
                continue
            cluster_nbrs[cb].add(cn)
            cluster_nbrs[cn].add(cb)

    cluster_nbrs_list = {k: sorted(v) for k, v in cluster_nbrs.items()}
    return cluster_preds_list, cluster_nbrs_list

def transitive_closure(preds: Dict[int, List[int]]) -> Dict[int, Set[int]]:
    """pred_closure[c] = all transitive predecessors of cluster c."""
    cache: Dict[int, Set[int]] = {}
    def dfs(c: int, visiting: Set[int]) -> Set[int]:
        if c in cache:
            return cache[c]
        out = set()
        for p in preds.get(c, []):
            if p in visiting:
                continue
            visiting.add(p)
            out.add(p)
            out |= dfs(p, visiting)
        cache[c] = out
        return out
    return {c: dfs(c, set()) for c in preds.keys()}


# ============================================================
# 5) Stockpiles + plant blending (strict window, no LP)
# ============================================================

@dataclass
class StockSystem:
    ranges: List[Tuple[float, float]]
    caps: List[float]
    low: float
    up: float
    cap: float
    cutoff: float

class Blender:
    """
    Deterministic, LP-free blender:
      - choose sources (mined ore bins + stockpiles) to fill up to plant cap
      - keep blended feed grade within [low, up]
      - send leftover mined ore to stockpiles by band with caps
    """
    def __init__(self, sys: StockSystem, n_bins: int = 10):
        self.sys = sys
        self.n_bins = int(n_bins)
        self.t = np.zeros(len(sys.ranges), float)
        self.m = np.zeros(len(sys.ranges), float)

    def reset_state(self):
        self.t[:] = 0.0
        self.m[:] = 0.0

    def _sp_grade(self, i: int) -> float:
        return 0.0 if self.t[i] <= 1e-12 else float(self.m[i] / self.t[i])

    def _which_pile(self, g: float) -> Optional[int]:
        for i, (a, b) in enumerate(self.sys.ranges):
            if a <= g < b:
                return i
        return None

    def _bin_mined(self, ore_pairs: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        if not ore_pairs:
            return np.zeros(self.n_bins), np.zeros(self.n_bins)
        gmax = max(1e-9, max(float(g) for _, g in ore_pairs))
        edges = np.linspace(0.0, gmax, self.n_bins + 1)
        t_bins = np.zeros(self.n_bins)
        m_bins = np.zeros(self.n_bins)
        for t, g in ore_pairs:
            t = float(t); g = float(g)
            if t <= 1e-12:
                continue
            k = np.searchsorted(edges, g, side="right") - 1
            k = min(max(k, 0), self.n_bins - 1)
            t_bins[k] += t
            m_bins[k] += t * g
        g_bins = np.divide(m_bins, np.maximum(1e-9, t_bins), out=np.zeros_like(t_bins), where=t_bins > 1e-12)
        return t_bins, g_bins

    def _window_fill(
        self,
        cap: float,
        low: float,
        up: float,
        src_t: np.ndarray,
        src_g: np.ndarray,
        prefer_order: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Fill up to cap with grade in [low, up], deterministic greedy.
        """
        K = len(src_t)
        take = np.zeros(K, float)
        PT = 0.0
        PM = 0.0

        # seed: try low+high mix if empty
        idx_av = np.where(src_t > 1e-12)[0]
        if idx_av.size > 0:
            iL = idx_av[np.argmin(src_g[idx_av])]
            iH = idx_av[np.argmax(src_g[idx_av])]
            gL, gH = float(src_g[iL]), float(src_g[iH])
            if (gL <= up + 1e-12) and (gH >= low - 1e-12) and (gH > gL + 1e-12):
                target = float(min(max(0.5*(low+up), gL + 1e-8), gH - 1e-8))
                r = (target - gL) / max(1e-12, (gH - target))  # xH = r*xL
                xL = min(src_t[iL], cap/(1.0 + r))
                xH = min(src_t[iH], r*xL)
                if xH < r*xL - 1e-12:
                    xL = xH / max(1e-12, r)
                take[iL] += xL; take[iH] += xH
                PT = float(take.sum())
                PM = float((take * src_g).sum())

        def x_max(i, PT, PM, cap_left):
            if src_t[i] <= take[i] + 1e-12 or cap_left <= 1e-12:
                return 0.0
            a = float(src_t[i] - take[i])
            g = float(src_g[i])
            if PT <= 1e-12:
                return min(a, cap_left)
            # want (PM + x g)/(PT + x) in [low, up]
            # => x constraints depend on g
            x_up = np.inf
            x_lo = np.inf
            if g > up + 1e-12:
                x_up = max(0.0, (up*PT - PM) / max(1e-12, (g - up)))
            if g < low - 1e-12:
                x_lo = max(0.0, (PM - low*PT) / max(1e-12, (low - g)))
            return float(max(0.0, min(a, cap_left, x_up, x_lo)))

        # greedy fill
        it = 0
        while PT < cap - 1e-9 and it < 20000:
            it += 1
            cap_left = cap - PT
            if PT <= 1e-12:
                PG = 0.5*(low+up)
            else:
                PG = PM / PT
            progressed = False

            # candidate list depending on current PG
            if PG < low - 1e-9:
                cand = [i for i in prefer_order if (src_t[i] > take[i] + 1e-12) and (src_g[i] > PG)]
            elif PG > up + 1e-9:
                cand = [i for i in prefer_order if (src_t[i] > take[i] + 1e-12) and (src_g[i] < PG)]
            else:
                cand_in = [i for i in prefer_order if (src_t[i] > take[i] + 1e-12) and (low - 1e-12 <= src_g[i] <= up + 1e-12)]
                cand = cand_in if cand_in else [i for i in prefer_order if (src_t[i] > take[i] + 1e-12)]

            for i in cand:
                x = x_max(int(i), PT, PM, cap_left)
                if x > 1e-9:
                    take[i] += x
                    PT += x
                    PM += x * float(src_g[i])
                    progressed = True
                    if PT >= cap - 1e-9:
                        break
            if not progressed:
                break

        PG = 0.0 if PT <= 1e-12 else PM / PT
        if not (low - 1e-6 <= PG <= up + 1e-6):
            return np.zeros_like(take), 0.0, 0.0
        # strict clamp
        if PT > cap + 1e-9:
            scale = cap / PT
            take *= scale
            PT = cap
            PG = 0.0 if PT <= 1e-12 else float((take * src_g).sum() / PT)
        return take, float(PT), float(PG)

    def blend_year(
        self,
        mined_ore_pairs: List[Tuple[float, float]],
        plant_target_grade: float,
        reclaim_bias: float,
    ) -> Dict[str, object]:
        """
        mined_ore_pairs = ore only (ton, grade) from this year (already cutoff-filtered by env)
        reclaim_bias > 0 -> prefer reclaim, < 0 -> prefer mined, 0 neutral.
        """
        low, up, cap = float(self.sys.low), float(self.sys.up), float(self.sys.cap)

        mined_t, mined_g = self._bin_mined(mined_ore_pairs)
        sp_t = self.t.copy()
        sp_g = np.array([self._sp_grade(i) for i in range(len(self.t))], float)

        # sources: [mined_bins || stockpiles]
        src_t = np.concatenate([mined_t, sp_t])
        src_g = np.concatenate([mined_g, sp_g])

        # preference scoring:
        #  - closer to target grade is better
        #  - reclaim_bias tilts stock sources vs mined sources
        tgt = float(plant_target_grade)
        dist = np.abs(src_g - tgt)
        score = 1.0 / (1e-6 + dist)

        nb = len(mined_t)
        if reclaim_bias > 0:
            score[nb:] *= (1.0 + 0.8 * reclaim_bias)
        elif reclaim_bias < 0:
            score[:nb] *= (1.0 + 0.8 * (-reclaim_bias))

        # only available sources
        score *= (src_t > 1e-12).astype(float)
        order = np.argsort(-score)

        take, PT, PG = self._window_fill(cap, low, up, src_t, src_g, order)

        # split
        mined_take = take[:nb]
        stock_take = take[nb:]

        # reclaim accounting (use pre-withdraw grades)
        pre_g = np.array([self._sp_grade(i) for i in range(len(self.t))], float)
        for i, x in enumerate(stock_take):
            if x > 1e-12:
                self.t[i] -= x
                self.m[i] -= x * pre_g[i]

        # mined->process
        mine_proc_t = float(mined_take.sum())
        mine_proc_m = float((mined_take * mined_g).sum())
        mine_proc_g = 0.0 if mine_proc_t <= 1e-12 else mine_proc_m / mine_proc_t

        # leftover mined -> stock by band
        sp_in_t = np.zeros(len(self.t), float)
        sp_in_m = np.zeros(len(self.t), float)
        sp_ex   = np.zeros(len(self.t), float)

        for k in range(nb):
            leftover = max(0.0, float(mined_t[k] - mined_take[k]))
            if leftover <= 1e-12:
                continue
            g = float(mined_g[k])
            j = self._which_pile(g)
            if j is None:
                continue
            space = float(self.sys.caps[j] - self.t[j])
            put = min(space, leftover)
            exc = max(0.0, leftover - put)
            if put > 1e-12:
                self.t[j] += put
                self.m[j] += put * g
                sp_in_t[j] += put
                sp_in_m[j] += put * g
            if exc > 1e-12:
                sp_ex[j] += exc

        # totals
        sp_out_t = stock_take
        sp_out_m = stock_take * pre_g
        PT_tot = float(mine_proc_t + sp_out_t.sum())
        PM_tot = float(mine_proc_m + sp_out_m.sum())
        PG_tot = 0.0 if PT_tot <= 1e-12 else PM_tot / PT_tot

        end_g = np.divide(self.m, np.maximum(1e-9, self.t), out=np.zeros_like(self.t), where=self.t > 1e-12)
        sp_out_g = np.where(sp_out_t > 1e-12, pre_g, 0.0)
        sp_in_g  = np.divide(sp_in_m, np.maximum(1e-9, sp_in_t), out=np.zeros_like(sp_in_t), where=sp_in_t > 1e-12)

        return dict(
            PT=float(PT_tot), PG=float(PG_tot),
            mine_to_proc_t=float(mine_proc_t), mine_to_proc_g=float(mine_proc_g),
            stock_to_proc_t=sp_out_t.tolist(), stock_to_proc_g=sp_out_g.tolist(),
            mine_to_stock_t=sp_in_t.tolist(),  mine_to_stock_g=sp_in_g.tolist(),
            stock_excess=sp_ex.tolist(),
            stock_end_t=self.t.tolist(), stock_end_g=end_g.tolist(),
        )


# ============================================================
# 6) Two-agent Environment (clusters as actions)
# ============================================================
def init_belief_from_clusters(clusters, n_scenarios: int, prior_sigma_rel: float, rng=None) -> BeliefState:
    """
    Initialise per-cluster grade scenarios around the cluster's current ore_grade.
    prior_sigma_rel is relative stdev (e.g., 0.10 means 10% of grade).
    """
    rng = np.random.default_rng(None if rng is None else rng)
    C = len(clusters)
    S = int(n_scenarios)
    base = np.array([float(c.ore_grade) for c in clusters], dtype=float)

    sigma = np.maximum(1e-6, prior_sigma_rel * np.maximum(1e-6, base))
    g_scen = rng.normal(loc=base[:, None], scale=sigma[:, None], size=(C, S))

    # keep grades non-negative
    g_scen = np.clip(g_scen, 0.0, None)

    b = BeliefState(cluster_grade_scen=g_scen)
    b.recompute_summaries()
    return b


class TwoAgentEnv:
    """
    One episode schedules all clusters (panels).
    Within each year:
      - plant agent selects mode (target grade + reclaim bias)
      - mining agent selects clusters; env enforces hard constraints and auto-fills

    Hard constraints:
      - precedence (always) via transitive closure mining package
      - mining capacity per year (always)
      - minimum mining width: implemented by panel actions + (if Condition=1) contiguity rules
      - exposed height / max allowable depth if indicator=1 (global exposed height <= H_max)
    """

    def __init__(
        self,
        cfg: Configuration,
        blocks_df: pd.DataFrame,
        clusters: List[Cluster],
        cluster_preds: Dict[int, List[int]],
        cluster_neighbors: Dict[int, List[int]],
        const: Dict[str, float],
    ):
        self.cfg = cfg
        self.blocks_df = blocks_df
        self.clusters = sorted(clusters, key=lambda c: c.cid)
        self.C = len(self.clusters)

        self.cid_to_idx = {c.cid: i for i, c in enumerate(self.clusters)}
        self.idx_to_cid = {i: c.cid for i, c in enumerate(self.clusters)}

        self.cluster_preds = {int(k): [int(x) for x in v] for k, v in cluster_preds.items()}
        self.cluster_neighbors = {int(k): [int(x) for x in v] for k, v in cluster_neighbors.items()}
        self.pred_closure = transitive_closure(self.cluster_preds)

        self.capacity = float(cfg.mining_capacity)
        self.process_cap = float(cfg.process_capacity)
        self.cutoff = float(cfg.cutoff)

        # exposed height constraint
        H_max = float(const.get("H_max", -1.0))
        self.use_height = (int(cfg.max_allowable_depth_priority_indicator) == 1) and (H_max > 0)
        self.H_max = H_max

        # min-width hard mode
        self.hard_min_width = (int(cfg.Condition) == 1)
        self.year_log = []   # list of dicts, one per committed year

        # stock system
        sys = StockSystem(
            ranges=list(cfg.stockpiles),
            caps=[float(x) for x in cfg.stockpiles_capacity],
            low=float(cfg.user_define_low_limit),
            up=float(cfg.user_define_up_limit),
            cap=float(cfg.process_capacity),
            cutoff=float(cfg.cutoff),
        )
        self.blender = Blender(sys)

        # Precompute cluster feature matrix
        self._build_features()
        # --- Belief state + observation buffer ---
        self.use_belief = bool(TWEAK.get("use_belief_state", False))
        self.obs_buffer = []  # incoming measurements: list[dict]
        # cluster_to_blocks exists implicitly; if you want explicit:
        self.cluster_to_blocks = {int(c.cid): list(c.blocks) for c in self.clusters}

        if self.use_belief:
            self.belief = init_belief_from_clusters(
                self.clusters,
                n_scenarios=int(TWEAK.get("n_scenarios", 20)),
                prior_sigma_rel=float(TWEAK.get("belief_prior_sigma_rel", 0.10)),
            )
        else:
            self.belief = None

        # for exposed height, we need per-block Z
        self._block_z = blocks_df.set_index("Indices")["Z"].to_dict()
        self.precedence_same_year_ok = (int(getattr(cfg, "precedence_same_year_ok", 1)) == 1)

        self.reset()

    def _build_features(self):
        # features per cluster (static)
        ton = np.array([c.tonnage for c in self.clusters], dtype=np.float32)
        val = np.array([c.value for c in self.clusters], dtype=np.float32)
        z   = np.array([c.z for c in self.clusters], dtype=np.float32)
        og  = np.array([c.ore_grade for c in self.clusters], dtype=np.float32)
        ot  = np.array([c.ore_tonnage for c in self.clusters], dtype=np.float32)

        val_d = val / np.maximum(ton, 1e-6)

        # downstream value proxy: count successors and their value
        succ = {c.cid: [] for c in self.clusters}
        for c, preds in self.cluster_preds.items():
            for p in preds:
                succ.setdefault(int(p), []).append(int(c))

        down_val = np.zeros(self.C, dtype=np.float32)
        down_cnt = np.zeros(self.C, dtype=np.float32)
        order = np.argsort(-z)  # top-down heuristic
        for i in order:
            cid = self.idx_to_cid[int(i)]
            for s in succ.get(cid, []):
                j = self.cid_to_idx[s]
                down_val[i] += val[j] + down_val[j]
                down_cnt[i] += 1.0 + down_cnt[j]

        def norm(a):
            a = a.astype(np.float32)
            m = float(np.mean(a))
            s = float(np.std(a))
            if s <= 1e-9:
                s = 1.0
            return (a - m) / s

        self.base_feats = np.stack([
            norm(ton),
            norm(val),
            norm(val_d),
            norm(z),
            norm(og),
            norm(ot),
            norm(down_val),
            norm(down_cnt),
            ], axis=1).astype(np.float32)  # [C,8]
        
        # Optional backward-compat alias:
        self.feats = self.base_feats
        
    def _belief_feats_local(self) -> np.ndarray:
        """
        Returns per-cluster belief summaries as features: [C, 4]
        (mu, sd, q10, q90) all z-scored across clusters.
        """
        if (not self.use_belief) or (self.belief is None):
            return np.zeros((self.C, 0), dtype=np.float32)

        b = self.belief
        b.recompute_summaries()
        mu = b.mu_grade.astype(np.float32)
        sd = b.sd_grade.astype(np.float32)
        q10 = b.q10_grade.astype(np.float32)
        q90 = b.q90_grade.astype(np.float32)

        def zscore(x):
            m = float(np.mean(x)); s = float(np.std(x))
            if s <= 1e-9: s = 1.0
            return (x - m) / s

        feat = np.stack([zscore(mu), zscore(sd), zscore(q10), zscore(q90)], axis=1).astype(np.float32)
        return feat  # [C,4]

    def _current_mining_feat_matrix(self) -> np.ndarray:
        """
        Returns [C, F] where F = 8 + 4 if belief features enabled.
        """
        if (not self.use_belief) or (not bool(TWEAK.get("belief_features_local", True))):
            return self.base_feats
        bfeat = self._belief_feats_local()  # [C,4]
        return np.concatenate([self.base_feats, bfeat], axis=1).astype(np.float32)

    # ------------------ plant observation & mode ------------------

    def plant_obs(self) -> np.ndarray:
        """
        Plant obs used by plant agent:
          [stock_fill..., stock_grade_norm..., year_norm, frac_mined, last_PG_norm, last_PT_frac]
        """
        N = len(self.blender.t)
        if N == 0:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        caps = np.array(self.blender.sys.caps, dtype=np.float32)
        t = np.array(self.blender.t, dtype=np.float32)
        g = np.zeros_like(t)
        nz = t > 1e-9
        g[nz] = (np.array(self.blender.m, dtype=np.float32)[nz] / (t[nz] + 1e-9))

        fill = t / (caps + 1e-9)
        low, up = float(self.blender.sys.low), float(self.blender.sys.up)
        g_norm = (g - low) / (up - low + 1e-9)

        year_norm = float(self.year) / 200.0
        frac_mined = float(self.mined_mask.sum()) / max(1.0, float(self.C))
        last_PG_norm = (float(self.last_PG) - low) / (up - low + 1e-9)
        last_PT_frac = float(self.last_PT) / (float(self.blender.sys.cap) + 1e-9)
        base = np.concatenate([fill, g_norm, np.array([year_norm, frac_mined, last_PG_norm, last_PT_frac], dtype=np.float32)])

        # --- global belief uncertainty metrics (optional) ---
        if self.use_belief and bool(TWEAK.get("belief_features_global", True)) and (self.belief is not None):
            self.belief.recompute_summaries()
            mean_sd = float(np.mean(self.belief.sd_grade))
            mean_iqr = float(np.mean(self.belief.q90_grade - self.belief.q10_grade))
            # normalise by plant window width for scale robustness
            win = float(up - low + 1e-9)
            extra = np.array([mean_sd / win, mean_iqr / win], dtype=np.float32)
            return np.concatenate([base, extra])

        return base

        

    def set_plant_mode(self, mode_idx: int):
        modes = TWEAK["plant_modes"]
        mode_idx = int(mode_idx) % len(modes)
        frac, reclaim_bias = modes[mode_idx]
        low, up = float(self.blender.sys.low), float(self.blender.sys.up)
        target = low + float(frac) * (up - low)
        self.plant_mode_idx = mode_idx
        self.plant_target_grade = float(target)
        self.plant_reclaim_bias = float(reclaim_bias)

    # ------------------ mining observation ------------------

    def mining_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          feats: [C, F]
          scalars: [S]
          mask: [C] feasible action mask
        scalars include year, remaining cap frac, mined frac, plant target norm, reclaim bias.
        """
        low, up = float(self.blender.sys.low), float(self.blender.sys.up)
        tgt_norm = (float(self.plant_target_grade) - low) / (up - low + 1e-9)

        max_seeds = float(TWEAK["max_face_seeds_per_year"])
        scalars = np.array([
            float(self.year) / 200.0,
            float(self.remaining) / (self.capacity + 1e-9),
            float(self.mined_mask.sum()) / max(1.0, float(self.C)),
            float(tgt_norm),
            float(self.plant_reclaim_bias),
        
            float(self.face_seeds_used) / (max_seeds + 1e-9),
            1.0 if getattr(self, "_using_lonely_seed_mode", False) else 0.0,
            1.0 if getattr(self, "lonely_seed_used_this_year", False) else 0.0,
            float(len(self.mined_clusters_this_year)) / (float(TWEAK.get("clusters_per_year_target", 12.0)) + 1e-9),
        ], dtype=np.float32)


        mask = self._feasible_mask(allow_lonely_seed=False)

        # ✅ deadlock fallback: if start-of-year strict mask empty, allow lonely seed
        self._using_lonely_seed_mode = False
        if self.hard_min_width and (not mask.any()) and (len(self.mined_clusters_this_year) == 0):
            mask2 = self._feasible_mask(allow_lonely_seed=True)
            if mask2.any():
                mask = mask2
                self._using_lonely_seed_mode = True
        
        feats = self._current_mining_feat_matrix()
        return feats, scalars, mask
        
    def _preds_satisfied(self, cid: int) -> bool:
        preds = self.cluster_preds.get(int(cid), [])
        for p in preds:
            ip = self.cid_to_idx[int(p)]
            if self.mined_mask[ip]:
                continue
            # allow predecessor in same year if enabled
            if self.precedence_same_year_ok and (int(p) in self.mined_clusters_this_year):
                continue
            return False
        return True

    # ------------------ constraints ------------------

    def _closure_unmined(self, cid: int) -> Set[int]:
        need = {int(cid)}
        for p in self.pred_closure.get(int(cid), set()):
            need.add(int(p))
    
        out = set()
        for c in need:
            i = self.cid_to_idx[int(c)]
            if self.mined_mask[i]:
                continue
            if int(c) in self.mined_clusters_this_year:   # <-- IMPORTANT
                continue
            out.add(int(c))
        return out


    def _closure_tonnage(self, closure: Set[int]) -> float:
        return float(sum(self.clusters[self.cid_to_idx[c]].tonnage for c in closure))

    def _closure_ore_pairs(self, closure: Set[int]) -> List[Tuple[float, float]]:
        pairs = []
        for c in closure:
            cl = self.clusters[self.cid_to_idx[c]]
            if cl.ore_tonnage > 1e-9 and cl.ore_grade >= self.cutoff:
                pairs.append((float(cl.ore_tonnage), float(cl.ore_grade)))
        return pairs

    def _height_ok_after_planning(self, planned_add: Set[int]) -> bool:
        """
        Global exposed height check:
          exposed = max Z among still-unmined blocks - min Z among blocks mined by year-end
        """
        if not self.use_height:
            return True

        # blocks mined by year-end = already mined clusters + planned clusters (this year)
        mined_clusters = set(self.mined_clusters_this_year) | set(self.mined_clusters_all_years) | set(planned_add)

        # compute z_low among mined blocks by year-end
        z_low = None
        for cid in mined_clusters:
            cl = self.clusters[self.cid_to_idx[cid]]
            z_low = cl.z if z_low is None else min(z_low, cl.z)

        if z_low is None:
            return True

        # unmined clusters after planning
        unmined_idx = np.where(~self.mined_mask)[0]
        # exclude clusters that are planned_add or already planned this year
        exclude = set(self.mined_clusters_this_year) | set(planned_add)
        z_top = None
        for i in unmined_idx:
            cid = self.idx_to_cid[int(i)]
            if cid in exclude:
                continue
            z_top = self.clusters[i].z if z_top is None else max(z_top, self.clusters[i].z)

        if z_top is None:
            # everything would be mined
            return True

        exposed = float(z_top - z_low)
        return exposed <= float(self.H_max) + 1e-6

    def _contiguity_ok(self, cid: int, allow_lonely_seed: bool = False) -> bool:
        if not self.hard_min_width:
            return True
    
        # Endgame: last remaining cluster, allow
        if int(self.mined_mask.sum()) == int(self.C - 1):
            return True
    
        # ---- Start of year (no planned clusters yet) ----
        if len(self.mined_clusters_this_year) == 0:
            # respect face seed budget
            if self.face_seeds_used >= int(TWEAK["max_face_seeds_per_year"]):
                return False
    
            # normal rule: must have a "real" neighbor package
            if self._seed_has_real_neighbor(cid):
                return True
    
            # fallback rule: allow lonely seed if explicitly permitted
            if allow_lonely_seed and bool(TWEAK.get("allow_lonely_seed_on_deadlock", True)):
                return True
    
            return False
    
        # ---- Mid-year: must touch current face OR start a new face if allowed ----
        nbrs = self.cluster_neighbors.get(int(cid), [])
        for n in nbrs:
            if int(n) in self.mined_clusters_this_year:
                return True
    
        # optionally allow multiple faces (new seeds)
        if bool(TWEAK["allow_multi_faces"]) and (self.face_seeds_used < int(TWEAK["max_face_seeds_per_year"])):
            if self._seed_has_real_neighbor(cid):
                return True
            if allow_lonely_seed and bool(TWEAK.get("allow_lonely_seed_on_deadlock", True)):
                return True
    
        return False

    

    def _seed_has_real_neighbor(self, cid: int) -> bool:
        for n in self.cluster_neighbors.get(int(cid), []):
            j = self.cid_to_idx[int(n)]
            if self.mined_mask[j] or (int(n) in self.mined_clusters_this_year):
                continue
            if not self._preds_satisfied(int(cid)):
                continue
            if not self._preds_satisfied(int(n)):
                continue
            ton_pair = float(self.clusters[self.cid_to_idx[int(cid)]].tonnage) + float(self.clusters[j].tonnage)
            if ton_pair <= self.remaining + 1e-9 and self._height_ok_after_planning({int(cid), int(n)}):
                return True
        return False


    def _feasible_mask(self, allow_lonely_seed: bool = False) -> np.ndarray:
        mask = np.zeros(self.C, dtype=bool)
        for i in range(self.C):
            if self.mined_mask[i]:
                continue
            cid = self.idx_to_cid[i]
            if int(cid) in self.mined_clusters_this_year:
                continue
    
            if not self._preds_satisfied(cid):
                continue
    
            if not self._contiguity_ok(cid, allow_lonely_seed=allow_lonely_seed):
                continue
    
            ton = float(self.clusters[i].tonnage)
            if ton > self.remaining + 1e-9:
                continue
    
            if not self._height_ok_after_planning({int(cid)}):
                continue
    
            mask[i] = True
        return mask

    def _capacity_rescue_fill(self) -> bool:
        """
        Try to fill remaining capacity by relaxing contiguity (lonely seeds allowed),
        but ONLY when we would otherwise waste a meaningful amount of capacity.
        Returns True if it added anything.
        """
        if not bool(TWEAK.get("capacity_rescue_enabled", True)):
            return False
    
        if self.remaining < float(TWEAK.get("capacity_rescue_min_unused_frac", 0.05)) * self.capacity:
            return False
    
        added_any = False
        added = 0
    
        while added < int(TWEAK.get("capacity_rescue_max_additions", 2000)):
            mask = self._feasible_mask(allow_lonely_seed=True)
            if not mask.any():
                break
    
            feas_idx = np.where(mask)[0]
    
            # Prefer "best fit" to use up remaining capacity
            best_i = None
            best_fit = 1e18
            for i in feas_idx:
                ton = float(self.clusters[i].tonnage)
                if ton <= self.remaining + 1e-9:
                    fit = abs(self.remaining - ton)
                    if fit < best_fit:
                        best_fit = fit
                        best_i = int(i)
    
            if best_i is None:
                break
    
            cid = int(self.idx_to_cid[best_i])
    
            # mark that we used lonely-seed mode (so reward can penalize)
            if self.hard_min_width and len(self.mined_clusters_this_year) == 0:
                self.lonely_seed_used_this_year = True
            # also treat any new face started in rescue as “lonely-ish”
            self.lonely_seed_used_this_year = True
    
            self._apply_mining_choice(cid, count_seed_if_new=True)
            added_any = True
            added += 1
    
            if self.remaining < float(TWEAK["autofill_min_remaining_frac"]) * self.capacity:
                break
    
        return added_any



    # ------------------ episode control ------------------

    def reset(self):
        self.year = 1
        self.remaining = float(self.capacity)
    
        self.mined_mask = np.zeros(self.C, dtype=bool)
        self.schedule_cluster_year = {}
    
        self.mined_clusters_all_years = set()
        self.mined_clusters_this_year = set()
    
        self.face_seeds_used = 0
        self.year_ore_pairs = []
    
        self.last_PT = 0.0
        self.last_PG = 0.0
    
        self.plant_mode_idx = 0
        self.plant_target_grade = float(self.blender.sys.low + 0.5*(self.blender.sys.up - self.blender.sys.low))
        self.plant_reclaim_bias = 0.0
    
        self.lonely_seed_used_this_year = False
        self.year_log = []
    
        # ✅ CRITICAL: reset stockpiles at the start of each episode
        self.blender.reset_state()
    
        return self.mining_obs()


    def _commit_year(self) -> float:
        """
        Apply year plan:
          - mark planned clusters mined, assign schedule year
          - run blender to compute plant outcomes and reward
          - roll to next year
        Returns: discounted year reward
        """
        pen_lonely = 0.0
        if getattr(self, "lonely_seed_used_this_year", False):
            pen_lonely = float(TWEAK.get("penalty_lonely_seed", 0.0))
        if len(self.mined_clusters_this_year) > 0:
            for cid in list(self.mined_clusters_this_year):
                i = self.cid_to_idx[cid]
                if not self.mined_mask[i]:
                    self.mined_mask[i] = True
                    self.schedule_cluster_year[int(cid)] = int(self.year)
                    self.mined_clusters_all_years.add(int(cid))

        # plant blending reward
        out = self.blender.blend_year(
            mined_ore_pairs=self.year_ore_pairs,
            plant_target_grade=float(self.plant_target_grade),
            reclaim_bias=float(self.plant_reclaim_bias),)
        ore_mined = sum(t for t, g in self.year_ore_pairs)
        ore_out = out["mine_to_proc_t"] + sum(out["mine_to_stock_t"]) + sum(out["stock_excess"])
        if abs(ore_mined - ore_out) > 1e-3:
            raise RuntimeError(f"ORE BALANCE FAIL year={self.year}: mined={ore_mined}, out={ore_out}")
        PT = float(out["PT"])
        PG = float(out["PG"])
        self.last_PT = PT
        self.last_PG = PG

        # economics: processed margin - rehandle - mining cost
        price = float(self.cfg.price)
        refc  = float(self.cfg.refining_cost)
        rec   = float(self.cfg.recovery) / 100.0
        pcost = float(self.cfg.p_cost)
        mcost = float(self.cfg.mining_cost)
        rhand = float(self.cfg.rehandling_cost)
        disc  = float(self.cfg.dis_rate) / 100.0

        # mining tonnage this year = sum tonnage of planned clusters
        mining_t = float(sum(self.clusters[self.cid_to_idx[c]].tonnage for c in self.mined_clusters_this_year))
        reclaimed_t = float(sum(out["stock_to_proc_t"])) if "stock_to_proc_t" in out else 0.0

        processed_margin = ((price - refc) * rec * PG - pcost) * PT
        rehandling_cost  = rhand * reclaimed_t
        mining_cost_term = mcost * mining_t

        # underuse penalties
        used_frac = 0.0 if self.capacity <= 1e-9 else min(1.0, mining_t / self.capacity)
        proc_frac = 0.0 if self.process_cap <= 1e-9 else min(1.0, PT / self.process_cap)

        pen_m = float(TWEAK["penalty_mining_underuse"]) * (max(0.0, 0.97 - used_frac) ** 2) * self.capacity
        pen_p = float(TWEAK["penalty_plant_underuse"]) * (max(0.0, 1.0 - proc_frac) ** 2) * self.process_cap

        cash_flow = processed_margin - rehandling_cost - mining_cost_term - pen_m - pen_p
        df = 1.0 / ((1.0 + disc) ** max(0, int(self.year) - 1))
        reward = float(cash_flow * df)
        
        committed_year = int(self.year)

        # compute ore/waste totals from planned clusters
        total_t = float(sum(self.clusters[self.cid_to_idx[c]].tonnage for c in self.mined_clusters_this_year))
        ore_t   = float(sum(self.clusters[self.cid_to_idx[c]].ore_tonnage for c in self.mined_clusters_this_year))
        waste_t = float(total_t - ore_t)
        ore_m   = float(sum(self.clusters[self.cid_to_idx[c]].ore_tonnage * self.clusters[self.cid_to_idx[c]].ore_grade
                           for c in self.mined_clusters_this_year))
        ore_g   = 0.0 if ore_t <= 1e-12 else ore_m / ore_t
        
        row = {
            "Period": committed_year,
            "Total Tonnage": total_t,
            "Waste Tonnage": waste_t,
            "Ore Tonnage": ore_t,
            "Ore Grade": ore_g,
            "Mine --> Process Tonnage": float(out["mine_to_proc_t"]),
            "Mine --> Process Grade": float(out["mine_to_proc_g"]),
            "Total Process Tonnage": float(out["PT"]),
            "Process Grade": float(out["PG"]),
            "NPV": float(cash_flow * df),   # or reward if you want to include penalties exactly
            "Plant mode idx": int(self.plant_mode_idx),
            "Plant target grade": float(self.plant_target_grade),
            "Reclaim bias": float(self.plant_reclaim_bias),
            "LonelySeedUsed": bool(getattr(self, "lonely_seed_used_this_year", False)),
        }
        
        # stockpile columns
        nsp = len(self.blender.t)
        for i in range(nsp):
            nm = f"Stock{i+1}"
            row[f"Mine --> {nm}(Tonnage)"] = float(out["mine_to_stock_t"][i])
            row[f"Mine --> {nm}(Grade)"] = float(out["mine_to_stock_g"][i])
            row[f"{nm}--> Process (Tonnage)"] = float(out["stock_to_proc_t"][i])
            row[f"{nm}--> Process (Grade)"] = float(out["stock_to_proc_g"][i])
            row[f"Mine --> {nm}(Tonnage) Stock Excess"] = float(out["stock_excess"][i])
            row[f"{nm}(Cummulative)"] = float(out["stock_end_t"][i])
            row[f"{nm}(Cummulative) Grade"] = float(out["stock_end_g"][i])
            row["Mining capacity"] = float(self.capacity)
            row["Mining utilisation"] = float(mining_t / (self.capacity + 1e-9))
            row["Plant capacity"] = float(self.process_cap)
            row["Plant utilisation"] = float(PT / (self.process_cap + 1e-9))
            row["Face seeds used"] = int(self.face_seeds_used)
            row["Lonely seed used"] = int(1 if pen_lonely > 0 else 0)
            row["Remaining capacity at commit"] = float(self.remaining)

        
        self.year_log.append(row)


        # reset year state
        self.year += 1
        if self.year > int(TWEAK["max_years_cap"]):
            # stop if absurd
            self.year = int(TWEAK["max_years_cap"])
        self.remaining = float(self.capacity)
        self.mined_clusters_this_year.clear()
        self.year_ore_pairs.clear()
        self.face_seeds_used = 0
        self.lonely_seed_used_this_year = False

        return reward
    
    def drain_stockpiles_post_mining(
        self,
        plant_net=None,          # optional: use plant policy to pick mode each tail year
        device=None,             # torch device if plant_net is used
        max_tail_years: int = 2000,
        min_process_t: float = 1e-6,
    ) -> int:
        """
        After ALL mining is finished, keep running the plant on reclaimed stock only.
        Appends rows to self.year_log with:
          - Total Tonnage = 0
          - Stocki --> Process filled by blender
        Stops when: no feasible in-window blend (PT ~ 0) OR no meaningful drain OR tail limit hit.
    
        Returns: number of tail years appended.
        """
        import numpy as np
        import math
    
        appended = 0
        disc = float(self.cfg.dis_rate) / 100.0
    
        # economics
        price = float(self.cfg.price)
        refc  = float(self.cfg.refining_cost)
        rec   = float(self.cfg.recovery) / 100.0
        pcost = float(self.cfg.p_cost)
        rhand = float(self.cfg.rehandling_cost)
    
        low, up = float(self.blender.sys.low), float(self.blender.sys.up)
    
        for _ in range(int(max_tail_years)):
            stock_before = float(np.sum(self.blender.t))
            if stock_before <= 1e-9:
                break
    
            # Optional: choose a plant mode each tail year from the plant policy
            if plant_net is not None:
                try:
                    import torch
                    if device is None:
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    po = torch.tensor(self.plant_obs(), dtype=torch.float32, device=device)
                    mode = int(torch.argmax(plant_net(po)).item())
                    self.set_plant_mode(mode)
                except Exception:
                    # fallback: keep current plant mode
                    pass
    
            out = self.blender.blend_year(
                mined_ore_pairs=[],
                plant_target_grade=float(self.plant_target_grade),
                reclaim_bias=float(self.plant_reclaim_bias),
            )
    
            PT = float(out["PT"])
            PG = float(out["PG"])
    
            stock_after = float(np.sum(self.blender.t))
            drained = stock_before - stock_after
    
            # Stop if blender can't make an in-window blend, or nothing meaningful drains
            if PT <= float(min_process_t) or drained <= 0.1 * float(min_process_t):
                break
            if not (low - 1e-9 <= PG <= up + 1e-9):
                break
    
            reclaimed_t = float(sum(out["stock_to_proc_t"]))  # all feed is reclaimed in tail
            processed_margin = ((price - refc) * rec * PG - pcost) * PT
            rehandling_cost  = rhand * reclaimed_t
    
            # Keep plant underuse penalty consistent with mining years (optional but consistent)
            proc_frac = 0.0 if self.process_cap <= 1e-9 else min(1.0, PT / self.process_cap)
            pen_p = float(TWEAK["penalty_plant_underuse"]) * (max(0.0, 1.0 - proc_frac) ** 2) * self.process_cap
    
            cash_flow = processed_margin - rehandling_cost - pen_p
    
            df = 1.0 / ((1.0 + disc) ** max(0, int(self.year) - 1))
            npv = float(cash_flow * df)
    
            row = {
                "Period": int(self.year),
                "Total Tonnage": 0.0,
                "Waste Tonnage": 0.0,
                "Ore Tonnage": 0.0,
                "Ore Grade": 0.0,
                "Mine --> Process Tonnage": 0.0,
                "Mine --> Process Grade": 0.0,
                "Total Process Tonnage": PT,
                "Process Grade": PG,
                "NPV": npv,
                "Plant mode idx": int(self.plant_mode_idx),
                "Plant target grade": float(self.plant_target_grade),
                "Reclaim bias": float(self.plant_reclaim_bias),
                "LonelySeedUsed": False,
            }
    
            nsp = len(self.blender.t)
            for i in range(nsp):
                nm = f"Stock{i+1}"
                row[f"Mine --> {nm}(Tonnage)"] = 0.0
                row[f"Mine --> {nm}(Grade)"] = 0.0
                row[f"{nm}--> Process (Tonnage)"] = float(out["stock_to_proc_t"][i])
                row[f"{nm}--> Process (Grade)"] = float(out["stock_to_proc_g"][i])
                row[f"Mine --> {nm}(Tonnage) Stock Excess"] = 0.0
                row[f"{nm}(Cummulative)"] = float(out["stock_end_t"][i])
                row[f"{nm}(Cummulative) Grade"] = float(out["stock_end_g"][i])
    
            self.year_log.append(row)
    
            # update state for next year
            self.last_PT = PT
            self.last_PG = PG
            self.year += 1
            appended += 1
    
        return appended


    def _autofill(self):
        """
        Greedy add-on clusters to fill remaining capacity, preserving constraints.
        """
        if not bool(TWEAK["autofill_enabled"]):
            return
        if self.remaining < float(TWEAK["autofill_min_remaining_frac"]) * self.capacity:
            return

        wv = float(TWEAK["autofill_score_w_value"])
        wg = float(TWEAK["autofill_score_w_grade"])
        wp = float(TWEAK["autofill_score_w_prox"])
        wf = float(TWEAK.get("autofill_score_w_fit", 0.0))
        target_left = 0.03 * self.capacity

        # compute proximity bonus to current planned face (centroid distance)
        planned = list(self.mined_clusters_this_year)
        if planned:
            P = np.array([[self.clusters[self.cid_to_idx[c]].cx, self.clusters[self.cid_to_idx[c]].cy] for c in planned], dtype=np.float64)
        else:
            P = None

        def prox_score(cid: int) -> float:
            if P is None:
                return 0.0
            c = self.clusters[self.cid_to_idx[cid]]
            d2 = np.sum((P - np.array([c.cx, c.cy]))**2, axis=1)
            d = float(np.sqrt(np.min(d2)))
            return 1.0 / (1e-6 + d)

        # grade desirability
        tgt = float(self.plant_target_grade)

        added = 0
        for _ in range(int(TWEAK["autofill_max_additions"])):
            mask = self._feasible_mask(allow_lonely_seed=True)
            if not mask.any():
                break
            # score feasible
            feas_idx = np.where(mask)[0]
            # skip if no room
            best = None
            best_s = -1e18
            for i in feas_idx:
                cid = self.idx_to_cid[int(i)]
                closure = self._closure_unmined(cid)
                ton = self._closure_tonnage(closure)
                if ton > self.remaining + 1e-9:
                    continue
                cl = self.clusters[i]
                vden = float(cl.value / max(1e-9, cl.tonnage))
                gscore = -abs(float(cl.ore_grade) - tgt)  # closer better
                pscore = prox_score(cid)
                fit = -abs((self.remaining - ton) - target_left) / (self.capacity + 1e-9)
                s = wv * vden + wg * gscore + wp * pscore + wf * fit
                if s > best_s:
                    best_s = s
                    best = cid
            if best is None:
                break
            # apply that best
            self._apply_mining_choice(best, count_seed_if_new=True)
            added += 1
            if self.remaining < float(TWEAK["autofill_min_remaining_frac"]) * self.capacity:
                break

    def _apply_mining_choice(self, cid: int, count_seed_if_new: bool):
        i = self.cid_to_idx[int(cid)]
        ton = float(self.clusters[i].tonnage)
        if ton > self.remaining + 1e-9:
            return
        if not self._height_ok_after_planning({int(cid)}):
            return
        if not self._preds_satisfied(cid):
            return
    
        # face seed counting unchanged (your logic ok)
        if self.hard_min_width and len(self.mined_clusters_this_year) == 0:
            if getattr(self, "_using_lonely_seed_mode", False):
                self.lonely_seed_used_this_year = True
    
        if self.hard_min_width:
            if len(self.mined_clusters_this_year) == 0:
                if count_seed_if_new:
                    self.face_seeds_used += 1
            else:
                nbrs = set(self.cluster_neighbors.get(int(cid), []))
                if not (nbrs & self.mined_clusters_this_year):
                    if count_seed_if_new:
                        self.face_seeds_used += 1
    
        self.mined_clusters_this_year.add(int(cid))
        self.remaining -= ton
    
        cl = self.clusters[i]
        if cl.ore_tonnage > 1e-9 and cl.ore_grade >= self.cutoff:
            self.year_ore_pairs.append((float(cl.ore_tonnage), float(cl.ore_grade)))


    def step_mining(self, action_idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], float, bool, Dict[str, object]]:
        """
        One mining-agent step.
        Returns reward only when year commits; otherwise small step reward.
        """
        info: Dict[str, object] = {}
        done = False

        feats, scalars, mask = self.mining_obs()
        # ✅ If nothing feasible, either commit the year (if we planned something),
        # or raise a REAL infeasibility at start-of-year.
        if not mask.any():
            # attempt rescue fill before committing
            rescued = self._capacity_rescue_fill()
            _, _, new_mask2 = self.mining_obs()   # <-- includes lonely-seed fallback logic
            if rescued and new_mask2.any():
                return self.mining_obs(), 0.0, False, {"rescued_capacity": True}
        
            info["year_committed"] = True
            committed_year = int(self.year)
            r = self._commit_year()
            info["committed_year"] = committed_year
            done = bool(self.mined_mask.all())
            return self.mining_obs(), float(r), done, info
           

       
        if action_idx < 0 or action_idx >= self.C or (not mask[action_idx]):
            info["invalid_action"] = True
            return (feats, scalars, mask), -0.01, False, info

        cid = self.idx_to_cid[int(action_idx)]
        # ✅ NEW safety
        if int(cid) in self.mined_clusters_this_year:
            info["already_planned"] = True
            return (feats, scalars, mask), -0.01, False, info
        allow_lonely = bool(getattr(self, "_using_lonely_seed_mode", False)) and (len(self.mined_clusters_this_year) == 0)
        if not self._contiguity_ok(cid, allow_lonely_seed=allow_lonely):
            info["contiguity_blocked"] = True
            return (feats, scalars, mask), -0.01, False, info

        closure = self._closure_unmined(cid)
        ton = self._closure_tonnage(closure)
        if ton > self.remaining + 1e-9:
            info["capacity_blocked"] = True
            return (feats, scalars, mask), -0.01, False, info

        if not self._height_ok_after_planning(closure):
            info["height_blocked"] = True
            return (feats, scalars, mask), -0.01, False, info

        # apply
        self._apply_mining_choice(cid, count_seed_if_new=True)

        # auto-fill to reduce "one block/year" and fill capacity
        self._autofill()

        # decide commit conditions:
        # - no feasible remaining OR remaining too small to fit any closure
        
        new_mask = self._feasible_mask()
        if not new_mask.any():
            info["year_committed"] = True
            committed_year = int(self.year)
            r = self._commit_year()
            info["committed_year"] = committed_year
            # done if all clusters mined
            if self.mined_mask.all():
                done = True
            # note: plant mode must be set again externally for next year
            return self.mining_obs(), float(r), done, info
        else:
            feas_idx = np.where(new_mask)[0]
            min_t = float(min(self.clusters[i].tonnage for i in feas_idx))
            if self.remaining + 1e-9 < min_t:
                info["year_committed"] = True
                committed_year = int(self.year)
                r = self._commit_year()
                info["committed_year"] = committed_year
                # done if all clusters mined
                if self.mined_mask.all():
                    done = True
                # note: plant mode must be set again externally for next year
                return self.mining_obs(), float(r), done, info

        # otherwise small shaping reward encouraging capacity packing
        pack = 1.0 - float(self.remaining) / (self.capacity + 1e-9)
        step_r = 0.001 * pack
        return self.mining_obs(), float(step_r), False, info

    def export_block_schedule(self) -> Dict[int, int]:
        """
        Convert cluster schedule -> block schedule.
        """
        sched: Dict[int, int] = {}
        for c in self.clusters:
            y = int(self.schedule_cluster_year.get(int(c.cid), int(self.year)))
            for b in c.blocks:
                sched[int(b)] = y
        return sched
    
    def export_report_df(self) -> pd.DataFrame:
        # self.year_log must exist; initialize it in reset()
        return pd.DataFrame(self.year_log)


# ============================================================
# 7) Networks (A2C): Mining actor-critic + Plant policy
# ============================================================

if _HAS_TORCH:
    class MiningActorCritic(nn.Module):
        """
        Improvements (points 3–7):
          (3) Actor is conditioned on scalars (year/remaining/target/etc.)
          (4) Critic pooling is mask-aware (pool only feasible clusters)
          (5) Critic also receives feasible-action fraction
          (6) Optional attention pooling (more expressive than plain mean)
          (7) Small stabilization: LayerNorm on encoded features
        """
        def __init__(self, feat_dim: int, scalar_dim: int, hid: int = 128, use_attention_pool: bool = True):
            super().__init__()
            self.use_attention_pool = bool(use_attention_pool)
    
            self.enc = nn.Sequential(
                nn.Linear(feat_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, hid),
                nn.ReLU(),
            )
            self.h_norm = nn.LayerNorm(hid)
    
            # (3) Actor conditioned on scalars
            self.actor = nn.Sequential(
                nn.Linear(hid + scalar_dim, hid),
                nn.ReLU(),
                nn.Linear(hid, 1),
            )
    
            # (6) Attention pooling for critic (optional)
            if self.use_attention_pool:
                self.att = nn.Linear(hid, 1)
    
            # (5) Critic takes pooled + scalars + feasible_frac (1 extra)
            self.critic = nn.Sequential(
                nn.Linear(hid + scalar_dim + 1, hid),
                nn.ReLU(),
                nn.Linear(hid, 1),
            )
    
        def forward(self, feats: torch.Tensor, scalars: torch.Tensor, mask: torch.Tensor | None = None):
            """
            feats:   [C, F]
            scalars: [S]
            mask:    [C] bool (feasible action mask). If None, treat all as feasible.
            """
            C = feats.shape[0]
            h = self.enc(feats)                 # [C, hid]
            h = self.h_norm(h)
    
            # ---- Actor (3): condition logits on scalars ----
            s_rep = scalars.unsqueeze(0).expand(C, scalars.shape[0])  # [C, S]
            logits = self.actor(torch.cat([h, s_rep], dim=1)).squeeze(-1)  # [C]
    
            # ---- Mask-aware pooling for critic (4,5,6) ----
            if mask is None:
                m = torch.ones(C, device=feats.device, dtype=torch.float32)
            else:
                m = mask.to(dtype=torch.float32)
    
            m_sum = m.sum()
            feasible_frac = (m_sum / float(C)).clamp(0.0, 1.0)
            
            denom = m_sum.clamp(min=1.0)   # only for safe division in pooling
    
            if self.use_attention_pool:
                # attention weights only over feasible
                att_logits = self.att(h).squeeze(-1)  # [C]
                neg_inf = torch.tensor(-1e9, device=feats.device, dtype=att_logits.dtype)
                att_logits = torch.where(m > 0.0, att_logits, neg_inf)
                w = torch.softmax(att_logits, dim=0)  # [C]
                pooled = (h * w.unsqueeze(1)).sum(dim=0)  # [hid]
            else:
                pooled = (h * m.unsqueeze(1)).sum(dim=0) / denom  # [hid]
    
            v_in = torch.cat([pooled, scalars, feasible_frac.unsqueeze(0)], dim=0)
            v = self.critic(v_in).squeeze(-1)
            return logits, v


    class PlantPolicy(nn.Module):
        def __init__(self, in_dim: int, n_modes: int, hid: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hid), nn.ReLU(),
                nn.Linear(hid, hid), nn.ReLU(),
                nn.Linear(hid, n_modes),
            )
            self.n_modes = n_modes

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

def _debug_deadlock(self):
    unmined = [self.idx_to_cid[int(i)] for i in np.where(~self.mined_mask)[0]]
    LOG.error(f"Unmined clusters: {unmined}")
    for cid in unmined:
        closure = self._closure_unmined(cid)
        ton = self._closure_tonnage(closure)
        cont = self._contiguity_ok(cid)
        h_ok = self._height_ok_after_planning(closure)
        LOG.error(f"cid={cid} closure_n={len(closure)} ton={ton:.2f} contiguity={cont} height_ok={h_ok}")
# ============================================================
# 8) Training loop
# ============================================================

def train_two_agent(env: TwoAgentEnv) -> Tuple[Dict[int, int], TwoAgentEnv]:
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available; cannot train RL.")
    LOG.info(f"RUNNING FILE: {__file__}")
    LOG.info(f"TWEAK epochs = {TWEAK.get('epochs')}, max_steps_per_episode = {TWEAK.get('max_steps_per_episode')}")
    # dims
    feats, scalars, mask = env.mining_obs()
    feat_dim = int(feats.shape[1])
    scalar_dim = int(scalars.shape[0])

    # plant dims
    plant_in = int(env.plant_obs().shape[0])
    n_modes = int(len(TWEAK["plant_modes"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MiningActorCritic(feat_dim, scalar_dim, hid=128).to(device)
    plant_net = PlantPolicy(plant_in, n_modes, hid=64).to(device)

    opt = optim.Adam(list(net.parameters()) + list(plant_net.parameters()), lr=float(TWEAK["lr"]))

    gamma = float(TWEAK["rl_gamma"])
    lam = float(TWEAK["gae_lambda"])
    ent_coef = float(TWEAK["entropy_coef"])

    for ep in range(1, int(TWEAK["epochs"]) + 1):
        # reset episode
        (feats_np, scalars_np, mask_np) = env.reset()

        # plant selects mode for year 1
        plant_logps = []
        plant_ents = []
        plant_rewards = []

        po = torch.tensor(env.plant_obs(), dtype=torch.float32, device=device)
        logits_p = plant_net(po)
        dist_p = torch.distributions.Categorical(logits=logits_p)
        mode = dist_p.sample()
        plant_logps.append(dist_p.log_prob(mode))
        plant_ents.append(dist_p.entropy())
        env.set_plant_mode(int(mode.item()))

        logps = []
        vals = []
        ents = []
        rews = []

        done = False
        steps = 0

        # track "current plant action" reward bucket per year
        current_year_reward = 0.0

        while not done and steps < int(TWEAK["max_steps_per_episode"]):
            steps += 1

            feats_t = torch.tensor(feats_np, dtype=torch.float32, device=device)
            scal_t = torch.tensor(scalars_np, dtype=torch.float32, device=device)
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device)

            logits, v = net(feats_t, scal_t, mask_t)

            # mask infeasible actions
            neg_inf = torch.tensor(-1e9, device=device, dtype=torch.float32)
            logits_masked = torch.where(mask_t, logits, neg_inf)

            dist = torch.distributions.Categorical(logits=logits_masked)
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy()

            (feats_np, scalars_np, mask_np), r, done, info = env.step_mining(int(a.item()))

            logps.append(logp)
            vals.append(v)
            ents.append(ent)
            rews.append(torch.tensor(r, dtype=torch.float32, device=device))

            current_year_reward += float(r)

            # if year committed, that reward belongs to the plant action of that year
            if info.get("year_committed", False):
                plant_rewards.append(torch.tensor(current_year_reward, dtype=torch.float32, device=device))
                current_year_reward = 0.0

                if not done:
                    # new plant action for next year
                    po = torch.tensor(env.plant_obs(), dtype=torch.float32, device=device)
                    logits_p = plant_net(po)
                    dist_p = torch.distributions.Categorical(logits=logits_p)
                    mode = dist_p.sample()
                    plant_logps.append(dist_p.log_prob(mode))
                    plant_ents.append(dist_p.entropy())
                    env.set_plant_mode(int(mode.item()))
            
        
        
        if len(rews) == 0:
            continue

        # ---- Mining A2C with GAE ----
        values = torch.stack(vals)              # [T]
        rewards = torch.stack(rews)             # [T]
        lps = torch.stack(logps)                # [T]
        ent = torch.stack(ents).mean()

        with torch.no_grad():
            next_v = torch.zeros(1, device=device)

        vs = torch.cat([values, next_v])
        adv = []
        gae = torch.zeros(1, device=device)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * vs[t+1] - vs[t]
            gae = delta + gamma * lam * gae
            adv.append(gae)
        advantages = torch.stack(list(reversed(adv))).detach().squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        returns = advantages + values.detach()

        policy_loss = -(lps * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss_m = policy_loss + value_loss - ent_coef * ent

        # ---- Plant REINFORCE on yearly rewards ----
        if len(plant_rewards) > 0:
            pr = torch.stack(plant_rewards)  # [Y]
            # discounted over years
            R = torch.zeros(1, device=device)
            ret_y = []
            for r in reversed(pr):
                R = r + gamma * R
                ret_y.append(R)
            ret_y = torch.stack(list(reversed(ret_y))).detach().squeeze(-1)
            ret_y = (ret_y - ret_y.mean()) / (ret_y.std(unbiased=False) + 1e-8)

            plps = torch.stack(plant_logps[:len(plant_rewards)])
            pent = torch.stack(plant_ents[:len(plant_rewards)]).mean()
            loss_p = -(plps * ret_y).mean() - (ent_coef * 0.5) * pent
        else:
            loss_p = torch.tensor(0.0, device=device)

        loss = loss_m + loss_p

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(net.parameters()) + list(plant_net.parameters()), 1.0)
        opt.step()

        if ep % 1 == 0:
            total_R = float(rewards.sum().item())
            LOG.info(f"[ep {ep:03d}] steps={steps:5d} return={total_R:,.2f} loss={float(loss.item()):.4f}")

    # ---- final greedy rollout ----
    # ---- final greedy rollout (robust) ----
    env.reset()
    done = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # first plant mode
    po = torch.tensor(env.plant_obs(), dtype=torch.float32, device=device)
    mode = torch.argmax(plant_net(po)).item()
    env.set_plant_mode(int(mode))
    
    greedy_steps = 0
    max_greedy_steps = 200000  # increase if you like, but with the guards below it should not hit
    
    while not done:
        greedy_steps += 1
        if greedy_steps > max_greedy_steps:
            raise RuntimeError(
                f"Greedy rollout exceeded max_greedy_steps. "
                f"year={env.year}, remaining={env.remaining:.3f}, "
                f"mined={int(env.mined_mask.sum())}/{env.C}"
            )
    
        feats_np, scalars_np, mask_np = env.mining_obs()
    
        # ✅ If nothing feasible, don't argmax into an invalid action.
        if not mask_np.any():
            # If we already planned something this year, just commit the year and continue.
            if len(env.mined_clusters_this_year) > 0:
                env._commit_year()
                done = bool(env.mined_mask.all())
                if not done:
                    po = torch.tensor(env.plant_obs(), dtype=torch.float32, device=device)
                    mode = torch.argmax(plant_net(po)).item()
                    env.set_plant_mode(int(mode))
                continue
    
            # Start-of-year and still nothing feasible => true infeasibility even after your fallback
            raise RuntimeError(
                "True infeasibility: no feasible clusters at start of year (even after fallback). "
                "Increase mining_capacity OR reduce closure size (split clusters / relax precedence / height)."
            )
    
        # normal greedy action selection WITH mask
        feats_t = torch.tensor(feats_np, dtype=torch.float32, device=device)
        scal_t  = torch.tensor(scalars_np, dtype=torch.float32, device=device)
        mask_t  = torch.tensor(mask_np, dtype=torch.bool, device=device)
    
        logits, _ = net(feats_t, scal_t, mask_t)
        logits_masked = torch.where(mask_t, logits, torch.tensor(-1e9, device=device))
        a = int(torch.argmax(logits_masked).item())

    
        (feats_np, scalars_np, mask_np), r, done, info = env.step_mining(a)
    
        # ✅ Safety: if something still goes wrong, pick any feasible action randomly
        if info.get("invalid_action", False):
            feas = np.where(mask_np)[0]
            if feas.size == 0:
                # if we planned something, commit; else hard fail
                if len(env.mined_clusters_this_year) > 0:
                    env._commit_year()
                    done = bool(env.mined_mask.all())
                    continue
                raise RuntimeError("Invalid action and no feasible actions: stuck.")
            a2 = int(np.random.choice(feas))
            (feats_np, scalars_np, mask_np), r, done, info = env.step_mining(a2)
    
        # if year committed, choose next plant mode
        if info.get("year_committed", False) and not done:
            po = torch.tensor(env.plant_obs(), dtype=torch.float32, device=device)
            mode = torch.argmax(plant_net(po)).item()
            env.set_plant_mode(int(mode))


    # ✅ POST-MINING: drain stockpiles into future years (append to env.year_log)
    try:
        env.drain_stockpiles_post_mining(plant_net=plant_net, device=device, max_tail_years=2000)
    except Exception as e:
        LOG.warning(f"Post-mining stock drain failed/stopped early: {e}")    
    schedule_blocks = env.export_block_schedule()
    return schedule_blocks, env


# ============================================================
# 9) Reporting/export (blocks schedule + plant simulation)
# ============================================================
def build_variables_sheet(cfg: Configuration, extras: dict | None = None) -> pd.DataFrame:
    """
    Build a Variables sheet containing ALL config keys from the TXT file (in file order),
    plus optional derived extras appended at the end.
    """
    rows = []

    # Prefer exact file contents if available
    if hasattr(cfg, "cfg_parsed") and isinstance(cfg.cfg_parsed, dict) and len(cfg.cfg_parsed) > 0:
        for k, v in cfg.cfg_parsed.items():
            rows.append({
                "Variable": str(k),
                "Value": v if not isinstance(v, (list, tuple, dict)) else json.dumps(v),
                "Type": type(v).__name__,
            })
    else:
        # Fallback: dump public attributes
        for k, v in sorted(cfg.__dict__.items()):
            if k.startswith("_"):
                continue
            if callable(v):
                continue
            rows.append({
                "Variable": str(k),
                "Value": v if not isinstance(v, (list, tuple, dict)) else json.dumps(v),
                "Type": type(v).__name__,
            })

    if extras:
        for k, v in extras.items():
            rows.append({
                "Variable": str(k),
                "Value": v if not isinstance(v, (list, tuple, dict)) else json.dumps(v),
                "Type": type(v).__name__,
            })

    return pd.DataFrame(rows)


def simulate_schedule_and_export(
    cfg: Configuration,
    blocks_df: pd.DataFrame,
    schedule_blocks: Dict[int, int],
    out_path: str,
):
    import ast
    import numpy as np
    import pandas as pd
    import math

    df = blocks_df.copy()

    # Ensure both "Value" and "Block value" exist if either exists
    if "Block value" not in df.columns and "Value" in df.columns:
        df["Block value"] = df["Value"]
    if "Value" not in df.columns and "Block value" in df.columns:
        df["Value"] = df["Block value"]

    # Use schedule column exactly as you want in Results sheet
    df["schedule"] = df["Indices"].astype(int).map(lambda b: int(schedule_blocks.get(int(b), 1))).astype(int)
    years = sorted(df["schedule"].unique().tolist())

    # --- Stock system (robust parsing if config stores strings) ---
    ranges = list(cfg.stockpiles) if not isinstance(cfg.stockpiles, str) else list(ast.literal_eval(cfg.stockpiles))
    caps = [float(x) for x in (cfg.stockpiles_capacity if not isinstance(cfg.stockpiles_capacity, str)
                              else ast.literal_eval(cfg.stockpiles_capacity))]

    sys = StockSystem(
        ranges=ranges,
        caps=caps,
        low=float(cfg.user_define_low_limit),
        up=float(cfg.user_define_up_limit),
        cap=float(cfg.process_capacity),
        cutoff=float(cfg.cutoff),
    )
    blender = Blender(sys)

    # economics
    price = float(cfg.price)
    refc  = float(cfg.refining_cost)
    rec   = float(cfg.recovery) / 100.0
    pcost = float(cfg.p_cost)
    disc  = float(cfg.dis_rate) / 100.0
    rhand = float(cfg.rehandling_cost)
    mcost = float(cfg.mining_cost)

    # --- output collectors ---
    rows = []
    recl_rows = []

    # Neutral plant mode for reporting (you can later swap in RL-mode values if you record them)
    plant_target = float(sys.low + 0.5 * (sys.up - sys.low))
    reclaim_bias = 0.0

    for y in years:
        yf = df[df["schedule"] == y]

        total_t = float(yf["Tonnage"].sum())
        ore_df = yf[yf["Grade"] >= float(cfg.cutoff)]
        ore_t = float(ore_df["Tonnage"].sum())
        waste_t = total_t - ore_t
        ore_g = 0.0 if ore_t <= 1e-12 else float((ore_df["Grade"] * ore_df["Tonnage"]).sum() / ore_t)

        ore_pairs = [(float(t), float(g)) for t, g in zip(ore_df["Tonnage"].to_numpy(), ore_df["Grade"].to_numpy())]
        out = blender.blend_year(ore_pairs, plant_target, reclaim_bias)

        stock_reclaim_t = float(sum(out["stock_to_proc_t"]))  # reclaimed from stockpiles
        mining_t = total_t                                    # ore+waste mined this period
        PT = float(out["PT"])
        PG = float(out["PG"])

        processed_margin = ((price - refc) * rec * PG - pcost) * PT
        rehandling_cost  = rhand * stock_reclaim_t
        mining_cost_term = mcost * mining_t
        cash_flow = processed_margin - rehandling_cost - mining_cost_term
        npv = cash_flow / ((1.0 + disc) ** (int(y) - 1))

        row = dict(
            Period=int(y),
            **{
                "Total Tonnage": total_t,
                "Waste Tonnage": waste_t,
                "Ore Tonnage": ore_t,
                "Ore Grade": ore_g,
                "Mine --> Process Tonnage": float(out["mine_to_proc_t"]),
                "Mine --> Process Grade": float(out["mine_to_proc_g"]),
                "Total Process Tonnage": float(out["PT"]),
                "Process Grade": float(out["PG"]),
                "NPV": float(npv),
            },
        )

        nsp = len(sys.ranges)
        for i in range(nsp):
            nm = f"Stock{i+1}"
            row[f"Mine --> {nm}(Tonnage)"] = float(out["mine_to_stock_t"][i])
            row[f"Mine --> {nm}(Grade)"] = float(out["mine_to_stock_g"][i])
            row[f"{nm}--> Process (Tonnage)"] = float(out["stock_to_proc_t"][i])
            row[f"{nm}--> Process (Grade)"] = float(out["stock_to_proc_g"][i])
            row[f"Mine --> {nm}(Tonnage) Stock Excess"] = float(out["stock_excess"][i])
            row[f"{nm}(Cummulative)"] = float(out["stock_end_t"][i])
            row[f"{nm}(Cummulative) Grade"] = float(out["stock_end_g"][i])

            recl_rows.append(dict(
                Year=int(y),
                Tonnage=float(out["stock_to_proc_t"][i]),
                Grade=float(out["stock_to_proc_g"][i]),
                StockpileNum=i + 1
            ))

        rows.append(row)

    # ---- Tail draining (optional but useful): process remaining stock after last mining year ----
    last_y = max(years) if years else 0
    tail_y = last_y
    tail_iter = 0
    MAX_TAIL = 1000
    eps_pt_tail = max(1e-6 * sys.cap, 1e-6)

    while True:
        stock_before = float(np.sum(blender.t))
        if stock_before <= 1e-6:
            break

        out = blender.blend_year([], plant_target, reclaim_bias)
        PT = float(out["PT"])
        PG = float(out["PG"])

        stock_after = float(np.sum(blender.t))
        drained = stock_before - stock_after

        if (PT < eps_pt_tail) or not (sys.low - 1e-9 <= PG <= sys.up + 1e-9) or (drained < 0.1 * eps_pt_tail):
            break

        tail_y += 1
        tail_iter += 1
        if tail_iter > MAX_TAIL:
            LOG.warning("Tail draining exceeded MAX_TAIL; stopping.")
            break

        n = int(tail_y) - 1
        dfac = math.exp(-n * math.log1p(disc))

        processed_margin = ((price - refc) * rec * PG - pcost) * PT
        rehandling_cost  = rhand * float(sum(out["stock_to_proc_t"]))
        cash_flow = processed_margin - rehandling_cost
        npv = cash_flow * dfac

        row = dict(
            Period=int(tail_y),
            **{
                "Total Tonnage": 0.0,
                "Waste Tonnage": 0.0,
                "Ore Tonnage": 0.0,
                "Ore Grade": 0.0,
                "Mine --> Process Tonnage": 0.0,
                "Mine --> Process Grade": 0.0,
                "Total Process Tonnage": PT,
                "Process Grade": PG,
                "NPV": float(npv),
            },
        )

        nsp = len(sys.ranges)
        for i in range(nsp):
            nm = f"Stock{i+1}"
            row[f"Mine --> {nm}(Tonnage)"] = 0.0
            row[f"Mine --> {nm}(Grade)"] = 0.0
            row[f"{nm}--> Process (Tonnage)"] = float(out["stock_to_proc_t"][i])
            row[f"{nm}--> Process (Grade)"] = float(out["stock_to_proc_g"][i])
            row[f"Mine --> {nm}(Tonnage) Stock Excess"] = 0.0
            row[f"{nm}(Cummulative)"] = float(out["stock_end_t"][i])
            row[f"{nm}(Cummulative) Grade"] = float(out["stock_end_g"][i])

            recl_rows.append(dict(
                Year=int(tail_y),
                Tonnage=float(out["stock_to_proc_t"][i]),
                Grade=float(out["stock_to_proc_g"][i]),
                StockpileNum=i + 1
            ))

        rows.append(row)

    report_df = pd.DataFrame(rows)
    reclaimed_df = pd.DataFrame(recl_rows)

    total_npv = float(report_df["NPV"].sum())
    print(f"Processed-only NPV (incl rehandling): {total_npv:,.2f}")

    variables_df = build_variables_sheet(
        cfg,
        extras={
            "Total Processed NPV (incl rehandling)": total_npv,
            "Resolved Num Stockpiles": int(len(sys.ranges)),
            "Reporting plant_target_grade": plant_target,
            "Reporting reclaim_bias": reclaim_bias,
        },
    )

    # Results sheet
    base_cols = ["Indices", "X", "Y", "Z", "Grade", "Tonnage", "schedule"]
    if "Block value" in df.columns:
        base_cols.insert(6, "Block value")
    results_df = df[base_cols].copy()

    out_path = cfg.output_file_name_path or out_path or "final_results.xlsx"
    with pd.ExcelWriter(out_path) as w:
        results_df.to_excel(w, sheet_name="Results", index=False)
        report_df.to_excel(w, sheet_name="Report", index=False)
        reclaimed_df.to_excel(w, sheet_name="Reclaimed Material", index=False)
        variables_df.to_excel(w, sheet_name="Variables", index=False)

    return report_df, out_path

def build_constraint_audit(env: TwoAgentEnv) -> pd.DataFrame:
    """
    Builds a year-by-year audit table based on env_final state + env_final.year_log.

    What it checks (at YEAR granularity):
      - Mining capacity used <= env.capacity
      - Process capacity used <= env.process_cap
      - Process grade within [low, up]
      - Stockpile cumulative tonnage <= stockpile cap
      - Cluster-level precedence satisfied by scheduled years
      - Contiguity components within allowed face seeds (approx)
      - Height exposed <= H_max (approx, using cluster Z)
    """
    import numpy as np
    import pandas as pd

    if not hasattr(env, "year_log") or not env.year_log:
        return pd.DataFrame()

    # ---- basic references ----
    low = float(env.blender.sys.low)
    up  = float(env.blender.sys.up)
    mine_cap = float(env.capacity)
    proc_cap = float(env.process_cap)

    # clusters lookup
    all_cids = [int(c.cid) for c in env.clusters]
    cid_to_cl = {int(c.cid): c for c in env.clusters}

    # year list from year_log (includes tail years)
    years = sorted({int(r.get("Period", 0)) for r in env.year_log if "Period" in r})
    if not years:
        return pd.DataFrame()

    # cluster schedule (mining years only)
    sched = getattr(env, "schedule_cluster_year", {}) or {}
    # invert: year -> list of cids
    year_to_cids = {}
    for cid, y in sched.items():
        year_to_cids.setdefault(int(y), []).append(int(cid))

    # precedence rule
    same_year_ok = bool(getattr(env, "precedence_same_year_ok", True))

    # contiguity knobs
    hard = bool(getattr(env, "hard_min_width", False))
    allow_multi_faces = bool(TWEAK.get("allow_multi_faces", True))
    max_seeds = int(TWEAK.get("max_face_seeds_per_year", 999999))

    # stockpile caps
    sp_caps = [float(x) for x in env.blender.sys.caps]
    nsp = len(sp_caps)

    # helper: connected components in planned clusters (using cluster_neighbors)
    def count_components(planned_set: set[int]) -> tuple[int, int]:
        if not planned_set:
            return 0, 0
        nbrs = env.cluster_neighbors
        seen = set()
        comps = 0
        lonely = 0
        for s in planned_set:
            if s in seen:
                continue
            comps += 1
            stack = [s]
            seen.add(s)
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for v in nbrs.get(int(u), []):
                    v = int(v)
                    if v in planned_set and v not in seen:
                        seen.add(v)
                        stack.append(v)
            if size == 1:
                lonely += 1
        return comps, lonely

    # helper: height exposed (approx)
    def exposed_height_after_year(y: int) -> float | None:
        if not bool(getattr(env, "use_height", False)):
            return None
        mined_up_to_y = {int(cid) for cid, yy in sched.items() if int(yy) <= int(y)}
        if not mined_up_to_y:
            return 0.0
        z_low = min(float(cid_to_cl[cid].z) for cid in mined_up_to_y)
        unmined = [cid for cid in all_cids if cid not in mined_up_to_y]
        if not unmined:
            return 0.0
        z_top = max(float(cid_to_cl[cid].z) for cid in unmined)
        return float(z_top - z_low)

    # ---- build audit rows ----
    audit_rows = []
    for y in years:
        # year_log row (for process checks, stock checks)
        yr = next((r for r in env.year_log if int(r.get("Period", -1)) == int(y)), {})
        PT = float(yr.get("Total Process Tonnage", 0.0))
        PG = float(yr.get("Process Grade", 0.0))

        # mining checks
        cids = year_to_cids.get(int(y), [])
        mining_t = float(sum(float(cid_to_cl[c].tonnage) for c in cids)) if cids else 0.0
        mining_ok = (mining_t <= mine_cap + 1e-6)

        proc_ok = (PT <= proc_cap + 1e-6)
        grade_ok = (low - 1e-9 <= PG <= up + 1e-9) if PT > 1e-9 else True  # no feed => skip

        # stock caps (use cumulative columns if present)
        stock_ok = True
        stock_max_over = 0.0
        for i in range(nsp):
            col = f"Stock{i+1}(Cummulative)"
            if col in yr:
                v = float(yr[col])
                over = v - sp_caps[i]
                if over > 1e-6:
                    stock_ok = False
                    stock_max_over = max(stock_max_over, over)

        # precedence violations (year-level)
        viol = 0
        sample = []
        for cid in cids:
            for p in env.cluster_preds.get(int(cid), []):
                p = int(p)
                py = sched.get(p, None)
                if py is None:
                    viol += 1
                    if len(sample) < 5:
                        sample.append(f"{cid}<-{p}(unscheduled)")
                    continue
                if same_year_ok:
                    bad = (int(py) > int(y))
                else:
                    bad = (int(py) >= int(y))
                if bad:
                    viol += 1
                    if len(sample) < 5:
                        sample.append(f"{cid}<-{p}(py={py})")

        precedence_ok = (viol == 0)

        # contiguity components (approx, only meaningful in hard mode)
        planned_set = set(int(x) for x in cids)
        comps, lonely_comps = count_components(planned_set)
        cont_ok = True
        if hard:
            # if multi-faces disabled => must be 1 component (unless empty)
            if not allow_multi_faces and comps > 1:
                cont_ok = False
            # if multi-faces enabled => components should not exceed seed budget
            if allow_multi_faces and comps > max_seeds:
                cont_ok = False

        # height exposed
        expH = exposed_height_after_year(int(y))
        height_ok = True
        if expH is not None:
            height_ok = (float(expH) <= float(env.H_max) + 1e-6)

        # lonely seed used flag (if you added it)
        lonely_flag = bool(yr.get("LonelySeedUsed", False))

        audit_rows.append({
            "Year": int(y),

            "MiningTonnage": mining_t,
            "MiningCap": mine_cap,
            "MiningCapOK": bool(mining_ok),

            "ProcessTonnage": PT,
            "ProcessCap": proc_cap,
            "ProcessCapOK": bool(proc_ok),

            "ProcessGrade": PG,
            "GradeLow": low,
            "GradeUp": up,
            "PlantGradeOK": bool(grade_ok),

            "StockCapOK": bool(stock_ok),
            "MaxStockOverCap": float(stock_max_over),

            "PrecedenceOK": bool(precedence_ok),
            "PrecedenceViolations": int(viol),
            "PrecedenceViolationExamples": "; ".join(sample),

            "ContiguityComponents": int(comps),
            "LonelyComponents": int(lonely_comps),
            "ContiguityOK": bool(cont_ok),

            "UseHeight": bool(getattr(env, "use_height", False)),
            "ExposedHeight": (None if expH is None else float(expH)),
            "H_max": (None if expH is None else float(env.H_max)),
            "HeightOK": bool(height_ok),

            "LonelySeedUsed": bool(lonely_flag),
        })

    return pd.DataFrame(audit_rows)


    


# ============================================================
# 10) Orchestrator
# ============================================================

def main(config_path: str):
    t0 = time.time()
    cfg = Configuration(config_path)

    LOG.info("Loading + preprocessing blocks...")
    df_raw = load_blocks_csv(cfg)
    blocks_df, precedence, adjacency, const = preprocess_blocks(df_raw, cfg)

    LOG.info("Building min-width clusters (panel actions)...")
    clusters, block_to_cluster = build_min_width_clusters(blocks_df, cfg, const)

    LOG.info("Building cluster precedence + adjacency graphs...")
    cluster_preds, cluster_nbrs = build_cluster_graph(clusters, block_to_cluster, precedence, adjacency)

    LOG.info("Creating two-agent environment...")
    env = TwoAgentEnv(cfg, blocks_df, clusters, cluster_preds, cluster_nbrs, const)

    feats, scalars, mask = env.mining_obs()
    pobs = env.plant_obs()
    
    print("Mining feats shape:", feats.shape)   # (C, 8) or (C, 12+)
    print("Mining scalars shape:", scalars.shape)
    print("Mining mask shape:", mask.shape)
    print("Plant obs shape:", pobs.shape)
    
    assert feats.shape[0] == env.C
    assert feats.shape[1] >= 8
    assert mask.shape == (env.C,)
    
    # belief-specific checks (optional but useful)
    if bool(TWEAK.get("use_belief_state", False)):
        assert feats.shape[1] > 8, "Belief enabled but mining feats did not expand."
        assert getattr(env, "belief", None) is not None, "Belief enabled but env.belief is None."
    LOG.info("Training two-agent RL (plant + mining)...")
    schedule_blocks, env_final = train_two_agent(env)

    # RL-consistent report
    report_df = env_final.export_report_df()
    audit_df = build_constraint_audit(env_final)
    env_final.audit_df = audit_df  # ✅ keep inside env object
    if audit_df.empty:
        raise RuntimeError("Audit is empty (env_final.year_log empty).")
    
    # ---- precedence (sum over all years) ----
    prec_viol = int(audit_df["PrecedenceViolations"].sum())
    if prec_viol > 0:
        ex = audit_df.loc[audit_df["PrecedenceViolations"] > 0, "PrecedenceViolationExamples"].head(5).tolist()
        raise RuntimeError(f"Precedence violations found: {prec_viol}. Examples: {ex}")
    
    # ---- capacity (max overshoot) ----
    max_mine_over = float((audit_df["MiningTonnage"] - audit_df["MiningCap"]).clip(lower=0.0).max())
    max_proc_over = float((audit_df["ProcessTonnage"] - audit_df["ProcessCap"]).clip(lower=0.0).max())
    if max(max_mine_over, max_proc_over) > 1e-6:
        raise RuntimeError(
            f"Capacity violation found. max_mine_over={max_mine_over:.3f}, max_proc_over={max_proc_over:.3f}"
        )
    
    # ---- height ----
    if env_final.use_height:
        height_viol_years = int((~audit_df["HeightOK"]).sum())
        if height_viol_years > 0:
            bad_years = audit_df.loc[~audit_df["HeightOK"], "Year"].tolist()[:20]
            raise RuntimeError(f"Height violations found in {height_viol_years} years. Example years: {bad_years}")

    pub_cols = [
    "Period","Total Tonnage","Ore Tonnage","Waste Tonnage","Ore Grade",
    "Total Process Tonnage","Process Grade","NPV",
    "Mining utilisation","Plant utilisation",
    "Face seeds used","Lonely seed used"]
    pub_table = report_df[ [c for c in pub_cols if c in report_df.columns] ].copy()
    
    latex_yearly = pub_table.to_latex(index=False, float_format="%.3f")
    latex_audit  = audit_df.to_latex(index=False, float_format="%.3f")
    
    with open("table_yearly.tex", "w", encoding="utf-8") as f:
        f.write(latex_yearly)
    with open("table_audit.tex", "w", encoding="utf-8") as f:
        f.write(latex_audit)

    # Build Results sheet (block schedule)
    df_out = blocks_df.copy()
    ids = df_out["Indices"].astype(int).tolist()
    missing = sorted(set(ids) - set(schedule_blocks.keys()))
    if missing:
        raise RuntimeError(f"Schedule missing {len(missing)} blocks. Example IDs: {missing[:20]}")
    
    df_out["schedule"] = df_out["Indices"].astype(int).map(schedule_blocks).astype(int)


    results_cols = ["Indices", "X", "Y", "Z", "Grade", "Tonnage", "schedule"]
    if "Block value" in df_out.columns:
        results_cols.insert(6, "Block value")
    results_df = df_out[results_cols].copy()

    out_path = cfg.output_file_name_path or "final_results.xlsx"
    LOG.info("Exporting Excel...")
    variables_df = build_variables_sheet(cfg, extras={
        "Report source": "env_final.year_log (RL-consistent)",
    })

    with pd.ExcelWriter(out_path) as w:
        results_df.to_excel(w, sheet_name="Results", index=False)
        report_df.to_excel(w, sheet_name="Report", index=False)
        audit_df.to_excel(w, sheet_name="Audit", index=False)     # ✅ NEW
        variables_df.to_excel(w, sheet_name="Variables", index=False)

    LOG.info(f"Done. Wrote: {out_path}")
    LOG.info(f"Elapsed seconds: {int(time.time() - t0)}")
    return schedule_blocks, report_df, out_path


if __name__ == "__main__":
    import argparse
    import sys

    DEFAULT_CONFIG = r"C:\Users\km923\.conda\envs\geol-env\example_config.txt"

    parser = argparse.ArgumentParser(
        description="Two-agent mine + plant RL scheduler"
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=DEFAULT_CONFIG,
        help="Path to config txt file. If omitted, DEFAULT_CONFIG is used."
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Show basic plots."
    )

    args = parser.parse_args()

    # Run
    main(args.config_path)

