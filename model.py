"""DAFlow v2.4.2: deterministic drift + conditional residual flow +
nonstationary prototype mass growth for temporal scRNA-seq forecasting.

Training uses prototype-level UOT pairs, a residual flow, direct multi-horizon
supervision, and a nonstationary growth field. Checkpoints saved by this
script can be reloaded with --checkpoint for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "new_data"
OUT = CKPT = REPORT = None  # set per dataset in run()
UNDERCOVERAGE_WEIGHT = 2.0
MASS_KL_WEIGHT = 0.1
MASS_SLOPE_WEIGHT = 0.01


@dataclass(frozen=True)
class Config:
    name: str
    matrix: Path
    metadata: Path | None
    time_column: str
    time_order: tuple
    train_tps: tuple
    test_tps: tuple
    counts: bool


CONFIGS = {
    "cd1": Config(
        "cd1", DATA / "GSE193346_CD1_embryonic.h5ad", None, "stage",
        ("E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5",
         "E15.5", "E16.5", "E17.5", "E18"),
        (0, 1, 2), tuple(range(3, 10)), False,
    ),
    "zebrafish": Config(
        "zebrafish",
        DATA / "zebrafish_embryonic/new_processed/two_forecasting-count_data-hvg.csv",
        DATA / "zebrafish_embryonic/new_processed/meta_data.csv",
        "stage.nice",
        ("A-HIGH", "B-OBLONG", "C-DOME", "D-30", "E-50", "F-S",
         "G-60", "H-75", "I-90", "J-B", "K-3S", "L-6S"),
        tuple(range(10)), (10, 11), True,
    ),
    "wot": Config(
        "wot",
        DATA / "Schiebinger2019/reduce_processed/three_forecasting-norm_data-hvg.csv",
        DATA / "Schiebinger2019/reduce_processed/three_forecasting-meta_data.csv",
        "day", tuple(float(i) for i in range(19)), tuple(range(16)), (16, 17, 18), False,
    ),
    "wot_early": Config(
        "wot_early",
        DATA / "Schiebinger2019/reduce_processed/three_forecasting-norm_data-hvg.csv",
        DATA / "Schiebinger2019/reduce_processed/three_forecasting-meta_data.csv",
        "day", tuple(float(i) for i in range(19)), tuple(range(6)), tuple(range(6, 19)), False,
    ),
    "vitrobetacell": Config(
        "vitrobetacell", DATA / "vitrobetacell/h5ad_x1", None,
        "Stage", ("S3c", "S4c", "S5c", "S6c"), (0, 1), (2, 3), True,
    ),
}

PAPER_LGPOT = {
    "zebrafish": {10: 33.74, 11: 36.35},
    "wot": {16: 15.24, 17: 15.95, 18: 16.17},
}


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(config: Config):
    if config.name == "cd1":
        return _load_cd1(config)
    if config.name == "vitrobetacell":
        return _load_vitrobetacell(config)
    return _load_csv(config)


def _load_cd1(config: Config):
    """Training-only HVG selection (2000 genes); the gene list is cached so
    that all methods are evaluated in an identical gene space."""
    gene_cache = ROOT / "cd1_vq_persistent_reports" / "cd1_training_only_hvg2000.csv"
    obj = ad.read_h5ad(config.matrix)
    stages = obj.obs[config.time_column].astype(str).to_numpy()
    if gene_cache.exists():
        genes = pd.read_csv(gene_cache, header=None)[0].astype(str).to_numpy()
    else:
        train_mask = np.isin(stages, np.asarray(config.time_order)[list(config.train_tps)])
        train = obj[train_mask].copy()
        sc.pp.highly_variable_genes(train, n_top_genes=2000, flavor="cell_ranger", subset=False)
        genes = train.var_names[train.var["highly_variable"].to_numpy()]
    if len(genes) != 2000:
        raise ValueError(f"Expected 2000 training-only HVGs, obtained {len(genes)}")
    obj = obj[:, genes]
    x = obj.X.toarray().astype(np.float32) if hasattr(obj.X, "toarray") else np.asarray(obj.X, np.float32)
    groups = [x[stages == stage] for stage in config.time_order]
    print("CD1 stage cells", {s: len(g) for s, g in zip(config.time_order, groups)}, flush=True)
    return groups


def _load_vitrobetacell(config: Config):
    """Same training-only HVG protocol for the in-vitro beta-cell data."""
    files = [config.matrix / f"GSE114412_x1_S{i}c.h5ad" for i in range(3, 7)]
    objs = [ad.read_h5ad(p) for p in files]
    gene_cache = ROOT / "cd1_vq_persistent_reports" / "vitrobetacell_training_only_hvg2000.csv"
    if gene_cache.exists():
        genes = pd.read_csv(gene_cache, header=None)[0].astype(str).to_numpy()
    else:
        train_raw = ad.concat(
            [ad.AnnData(o.raw.X.copy(), var=pd.DataFrame(index=o.raw.var_names.copy()))
             for o in objs[:2]], axis=0, join="inner", merge="same"
        )
        try:
            sc.pp.highly_variable_genes(train_raw, n_top_genes=2000, flavor="seurat_v3", subset=False)
        except (ImportError, ValueError):
            sc.pp.highly_variable_genes(train_raw, n_top_genes=2000, flavor="cell_ranger", subset=False)
        genes = train_raw.var_names[train_raw.var["highly_variable"].to_numpy()]
    if len(genes) != 2000:
        raise ValueError(f"Expected 2000 training-only HVGs, obtained {len(genes)}")
    groups = []
    for o in objs:
        x = ad.AnnData(o.raw[:, genes].X.copy(), var=pd.DataFrame(index=genes.copy()))
        sc.pp.normalize_total(x, target_sum=1e4)
        sc.pp.log1p(x)
        groups.append(x.X.toarray().astype(np.float32) if hasattr(x.X, "toarray")
                      else np.asarray(x.X, dtype=np.float32))
    print("vitrobetacell stage cells", {s: len(g) for s, g in zip(config.time_order, groups)}, flush=True)
    return groups


def _load_csv(config: Config):
    """Loader for csv-backed datasets (zebrafish, WOT)."""
    frame = pd.read_csv(config.matrix, index_col=0)
    meta = pd.read_csv(config.metadata, index_col=0).loc[frame.index]
    x = frame.to_numpy(dtype=np.float32, copy=True)
    if config.counts:
        library = x.sum(1, keepdims=True)
        x = np.log1p(1e4 * x / np.maximum(library, 1.0)).astype(np.float32)
    mapping = {label: i for i, label in enumerate(config.time_order)}
    t = meta[config.time_column].map(mapping).to_numpy()
    if np.isnan(t).any():
        raise ValueError(f"Unmapped time labels in {config.name}")
    groups = [x[t == i] for i in range(len(config.time_order))]
    del frame
    print(config.name, "stage cells", {s: len(g) for s, g in zip(config.time_order, groups)}, flush=True)
    return groups


class AutoEncoder(nn.Module):
    def __init__(self, genes=2000, latent=32, hidden=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(genes, hidden), nn.SiLU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, latent), nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128), nn.SiLU(), nn.Linear(128, hidden), nn.SiLU(),
            nn.Linear(hidden, genes), nn.Softplus(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def train_ae(model, groups, train_tps, device, epochs, seed, batch=256):
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    steps = max(80, sum(len(groups[t]) for t in train_tps) // batch)
    history = []
    for epoch in range(epochs):
        losses = []
        model.train()
        for _ in range(steps):
            tp = int(rng.choice(train_tps))
            cells = groups[tp]
            idx = rng.choice(len(cells), min(batch, len(cells)), replace=len(cells) < batch)
            xb = torch.from_numpy(cells[idx]).to(device)
            loss = F.mse_loss(model(xb), xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        row = {"epoch": epoch + 1, "mse": float(np.mean(losses))}
        history.append(row)
        print("AE", row, flush=True)
    return history


def encode_groups(model, groups, device, batch=512):
    encoded = []
    model.eval()
    with torch.no_grad():
        for x in groups:
            blocks = []
            for start in range(0, len(x), batch):
                blocks.append(model.encode(torch.from_numpy(x[start:start + batch]).to(device)).cpu().numpy())
            encoded.append(np.concatenate(blocks).astype(np.float32))
    return encoded


def decode(model, z, device, batch=512):
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(z), batch):
            values.append(model.decode(torch.from_numpy(z[start:start + batch]).to(device)).cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def ot_dist_sq(a, b):
    a2 = (a * a).sum(1, keepdims=True)
    b2 = (b * b).sum(1, keepdims=True).T
    return np.maximum(a2 + b2 - 2.0 * a @ b.T, 0.0)


class PairModel:
    def __init__(self, t, target_t, z0, z1, km0, km1, gamma, centers0, centers1,
                 source_mass, target_mass, transition_prob, undercoverage,
                 oracle_mass_js):
        self.t = t
        self.target_t = target_t
        self.horizon = target_t - t
        self.z0 = z0
        self.z1 = z1
        self.km0 = km0
        self.km1 = km1
        self.gamma = gamma
        self.centers0 = centers0
        self.centers1 = centers1
        self.source_mass = source_mass
        self.target_mass = target_mass
        self.transition_prob = transition_prob
        self.undercoverage = undercoverage
        self.oracle_mass_js = oracle_mass_js


def prototype_mass_oracle_js(source_mass, transition_prob, target_mass,
                             steps=500, learning_rate=0.1):
    """Best reachable composition JS when only prototype weights may change."""
    logits = nn.Parameter(torch.log(torch.from_numpy(source_mass).float().clamp_min(1e-8)))
    transition = torch.from_numpy(transition_prob).float()
    target = torch.from_numpy(target_mass).float()
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    best = float("inf")
    for _ in range(steps):
        q = torch.softmax(logits, dim=0)
        loss = js_divergence(q @ transition, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        best = min(best, float(loss.detach().item()))
    return best


def build_pair(z0, z1, t, target_t, seed, n_proto=64):
    import ot as ot_lib
    k0 = min(n_proto, len(z0))
    k1 = min(n_proto, len(z1))
    pair_seed = seed + 97 * t + 1009 * (target_t - t)
    km0 = MiniBatchKMeans(k0, random_state=pair_seed, batch_size=1024, n_init=3).fit(z0)
    km1 = MiniBatchKMeans(k1, random_state=pair_seed + 1, batch_size=1024, n_init=3).fit(z1)
    active0 = np.unique(km0.labels_)
    active1 = np.unique(km1.labels_)
    c0 = km0.cluster_centers_[active0].astype(np.float64)
    c1 = km1.cluster_centers_[active1].astype(np.float64)
    km0.labels_ = np.searchsorted(active0, km0.labels_)
    km1.labels_ = np.searchsorted(active1, km1.labels_)
    k0, k1 = len(active0), len(active1)
    a = np.bincount(km0.labels_, minlength=k0).astype(np.float64); a /= a.sum()
    b = np.bincount(km1.labels_, minlength=k1).astype(np.float64); b /= b.sum()
    cost = ot_lib.dist(c0, c1, metric="sqeuclidean")
    cost /= max(float(np.median(cost[cost > 0])), 1e-8)
    gamma = ot_lib.unbalanced.sinkhorn_knopp_unbalanced(
        a, b, cost, reg=0.08, reg_m=1.0, numItermax=3000
    )
    gamma = np.maximum(gamma, 0)
    incoming = gamma.sum(0)
    incoming_share = incoming / max(float(incoming.sum()), 1e-12)
    undercoverage = np.clip(
        (b - incoming_share) / np.maximum(b, 1e-12), 0.0, 1.0
    ).astype(np.float32)
    gamma /= max(float(gamma.sum()), 1e-12)
    transition_prob = gamma / np.maximum(gamma.sum(1, keepdims=True), 1e-12)
    oracle_mass_js = prototype_mass_oracle_js(a, transition_prob, b)
    column_max = transition_prob.max(0)
    print("PAIR_MASS", {
        "source_t": t, "target_t": target_t,
        "weighted_undercoverage": float(np.sum(b * undercoverage)),
        "max_undercoverage": float(undercoverage.max()),
        "oracle_mass_js": oracle_mass_js,
        "min_column_max_transition": float(column_max.min()),
        "near_unreachable_target_mass": float(b[column_max < 1e-6].sum()),
    }, flush=True)
    return PairModel(t, target_t, z0, z1, km0, km1, gamma,
                     c0.astype(np.float32), c1.astype(np.float32),
                     a.astype(np.float32), b.astype(np.float32),
                     transition_prob.astype(np.float32), undercoverage,
                     oracle_mass_js)


def build_dynamics(encoded, train_tps, seed, max_horizon=3):
    pairs = []
    for horizon in range(1, max_horizon + 1):
        for index in range(len(train_tps) - horizon):
            source_t = train_tps[index]
            target_t = train_tps[index + horizon]
            pairs.append(build_pair(encoded[source_t], encoded[target_t],
                                    source_t, target_t, seed))
    return pairs


def sample_pair_cells(pair, n, rng):
    flat = rng.choice(pair.gamma.size, n, p=pair.gamma.ravel())
    p0, p1 = np.unravel_index(flat, pair.gamma.shape)
    members0 = [np.flatnonzero(pair.km0.labels_ == k) for k in range(len(pair.centers0))]
    members1 = [np.flatnonzero(pair.km1.labels_ == k) for k in range(pair.gamma.shape[1])]
    i0 = np.array([rng.choice(members0[k]) for k in p0])
    i1 = np.array([rng.choice(members1[k]) for k in p1])
    return pair.z0[i0], pair.z1[i1], p0, p1


def time_features(t):
    t = torch.as_tensor(t, dtype=torch.float32)
    if t.dim() == 0:
        t = t[None]
    if t.dim() == 1:
        t = t[:, None]
    return torch.cat([t, torch.sin(math.pi * t), torch.cos(math.pi * t)], 1)


class PriorNet(nn.Module):
    """Conditional mean displacement, explicitly conditioned on horizon."""

    def __init__(self, latent=32, gp_dim=32, emb_dim=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + gp_dim + emb_dim + 6, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, source, emb, bio_time, delta_time):
        # Keep the zero GP channel to load the original 78-input weight matrix.
        return self.net(torch.cat([
            source, torch.zeros_like(source), emb,
            time_features(bio_time), time_features(delta_time)
        ], 1))


class UnifiedFlow(nn.Module):
    """State flow plus a bounded nonstationary relative-growth readout."""

    def __init__(self, latent=32, gp_dim=32, emb_dim=8, hidden=256):
        super().__init__()
        self.latent = latent
        self.net = nn.Sequential(
            nn.Linear(2 * latent + gp_dim + emb_dim + 9, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent),
        )
        self.growth_readout = nn.Sequential(
            nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        self.growth_slope = nn.Sequential(
            nn.Linear(latent, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(self.growth_slope[-1].weight)
        nn.init.zeros_(self.growth_slope[-1].bias)

    def forward(self, residual, source, emb, path_time, bio_time, delta_time):
        # Keep the zero GP channel to load the original 113-input weight matrix.
        return self.net(torch.cat([
            residual, source, torch.zeros_like(source), emb,
            time_features(path_time), time_features(bio_time),
            time_features(delta_time)
        ], 1))

    def growth_rate(self, source, biological_time):
        base = self.growth_readout(source).squeeze(1)
        slope = self.growth_slope(source).squeeze(1)
        time = torch.as_tensor(biological_time, dtype=source.dtype, device=source.device)
        return base + torch.tanh(time) * slope, slope


def mmd2_rbf(x, y):
    """Biased MMD^2 with median-heuristic RBF bandwidth (per batch)."""
    z = torch.cat([x, y], 0)
    d2 = torch.cdist(z, z).square()
    med = d2[d2 > 0].median()
    g = 1.0 / (med + 1e-8)
    k = torch.exp(-g * d2)
    n = len(x)
    kxx = k[:n, :n].mean()
    kyy = k[n:, n:].mean()
    kxy = k[:n, n:].mean()
    return kxx + kyy - 2.0 * kxy


def euler_sample(flow, source, emb, bio_time, delta_time, steps):
    """Differentiable few-step Euler sample of the normalized innovation."""
    batch = len(source)
    r = torch.randn(batch, source.shape[1], device=source.device)
    dt = 1.0 / steps
    for i in range(steps):
        s = torch.full((batch, 1), (i + 0.5) * dt, device=source.device)
        output = flow(r, source, emb, s, bio_time, delta_time)
        r = r + dt * output[:, :source.shape[1]]
    return r


def growth_log_rate(flow, source, biological_time):
    """Bounded nonstationary propensity g0(z) + tanh(t) g1(z)."""
    return flow.growth_rate(source, biological_time)


def weighted_mse(pred, target, weight):
    per_cell = (pred - target).square().mean(1)
    return torch.sum(per_cell * weight) / torch.clamp(weight.sum(), min=1e-8)


def js_divergence(p, q):
    p = p.clamp_min(1e-8); p = p / p.sum()
    q = q.clamp_min(1e-8); q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (torch.sum(p * torch.log(p / m)) + torch.sum(q * torch.log(q / m)))


def categorical_kl(p, q):
    p = p.clamp_min(1e-8); p = p / p.sum()
    q = q.clamp_min(1e-8); q = q / q.sum()
    return torch.sum(p * torch.log(p / q))


def train_unified(prior, flow, embedding, log_sigma1, pairs, last_train_tp,
                  device, epochs, seed, args):
    rng = np.random.default_rng(seed)
    params = (list(prior.parameters()) + list(flow.parameters())
              + list(embedding.parameters()) + [log_sigma1])
    optimizer = torch.optim.AdamW(params, lr=5e-4, weight_decay=1e-5)
    history = []
    pairs_by_horizon = {
        horizon: [p for p in pairs if p.horizon == horizon]
        for horizon in sorted({p.horizon for p in pairs})
    }
    configured_weights = [float(x) for x in args.horizon_weights.split(",")]
    available_horizons = list(pairs_by_horizon)
    horizon_prob = np.asarray([
        configured_weights[h - 1] if h <= len(configured_weights) else 0.0
        for h in available_horizons
    ], dtype=np.float64)
    if horizon_prob.sum() <= 0:
        raise ValueError("At least one available horizon must have positive training weight")
    horizon_prob /= horizon_prob.sum()
    n_steps = max(80, 12 * len(pairs_by_horizon[1]))

    def sample_transition(source, source_t, horizon, steps):
        """Differentiable direct h-step model sample used by consistency loss."""
        batch_size = len(source)
        bt = torch.full((batch_size, 1), source_t / last_train_tp, device=device)
        delta = torch.full((batch_size, 1), horizon / last_train_tp, device=device)
        emb = embedding.weight.expand(batch_size, -1)
        sigma = torch.exp(log_sigma1).clamp(0.05, 5.0)
        correction = prior(source, emb, bt, delta)
        eps = euler_sample(flow, source, emb, bt, delta, steps=steps)
        return source + correction + sigma * eps

    for epoch in range(epochs):
        losses = {"loss": [], "flow": [], "prior": [], "mmd": [],
                  "consistency": [], "mass": [], "mass_js": [],
                  "mass_kl": [], "mass_slope": [],
                  "mass_ess_ratio": [], "undercoverage": []}
        sampled_horizons = []
        for step_index in range(n_steps):
            horizon = int(rng.choice(available_horizons, p=horizon_prob))
            candidates = pairs_by_horizon[horizon]
            pair = candidates[int(rng.integers(len(candidates)))]
            sampled_horizons.append(horizon)
            a, b, _, p1 = sample_pair_cells(pair, args.batch, rng)
            v_star = b - a
            sample_weight = 1.0 + UNDERCOVERAGE_WEIGHT * pair.undercoverage[p1]
            # Jitter lets the networks see biological times beyond training pairs.
            bt_cond = pair.t / last_train_tp + float(rng.uniform(0.0, args.bt_jitter))
            bt = np.full((args.batch, 1), bt_cond, np.float32)
            delta = np.full((args.batch, 1), pair.horizon / last_train_tp, np.float32)

            a_t = torch.from_numpy(a).to(device)
            b_t = torch.from_numpy(b).to(device)
            v_t = torch.from_numpy(v_star).to(device)
            bt_t = torch.from_numpy(bt).to(device)
            delta_t = torch.from_numpy(delta).to(device)
            sample_weight_t = torch.from_numpy(sample_weight.astype(np.float32)).to(device)
            emb = embedding.weight.expand(len(a_t), -1)
            sigma = torch.exp(log_sigma1).clamp(0.05, 5.0)

            correction = prior(a_t, emb, bt_t, delta_t)
            v_mean = correction
            eps_star = (v_t - v_mean.detach()) / sigma

            noise = torch.randn_like(eps_star)
            s = torch.rand(args.batch, 1, device=device)
            r = (1 - s) * noise + s * eps_star
            target = eps_star - noise
            pred_output = flow(r, a_t, emb, s, bt_t, delta_t)
            pred = pred_output[:, :a_t.shape[1]]
            loss_flow = weighted_mse(pred, target, sample_weight_t)
            loss_prior = weighted_mse(correction, v_t, sample_weight_t)

            if args.mmd_weight > 0:
                eps_gen = euler_sample(flow, a_t, emb, bt_t, delta_t,
                                       steps=args.mmd_steps)
                z_gen = a_t + v_mean + sigma * eps_gen
                loss_mmd = mmd2_rbf(z_gen, b_t)
            else:
                loss_mmd = torch.zeros((), device=device)

            loss_consistency = torch.zeros((), device=device)
            do_consistency = (args.consistency_weight > 0
                              and 2 in pairs_by_horizon
                              and step_index % args.consistency_every == 0)
            if do_consistency:
                pair2 = pairs_by_horizon[2][int(rng.integers(len(pairs_by_horizon[2])))]
                n_consistency = min(args.consistency_batch, args.batch)
                a2, _, _, _ = sample_pair_cells(pair2, n_consistency, rng)
                a2_t = torch.from_numpy(a2).to(device)
                direct = sample_transition(a2_t, pair2.t, 2, args.mmd_steps)
                composed = sample_transition(a2_t, pair2.t, 1, args.mmd_steps)
                composed = sample_transition(composed, pair2.t + 1, 1, args.mmd_steps)
                loss_consistency = mmd2_rbf(direct, composed)

            source_proto = torch.from_numpy(pair.centers0).to(device)
            source_mass = torch.from_numpy(pair.source_mass).to(device)
            growth_time = pair.t / max(last_train_tp, 1)
            log_rate, growth_slope = growth_log_rate(
                flow, source_proto, growth_time
            )
            q = torch.softmax(
                torch.log(source_mass.clamp_min(1e-8))
                + pair.horizon * log_rate,
                dim=0,
            )
            transition = torch.from_numpy(pair.transition_prob).to(device)
            target_mass = torch.from_numpy(pair.target_mass).to(device)
            predicted_mass = q @ transition
            loss_mass_js = js_divergence(predicted_mass, target_mass)
            loss_mass_kl = categorical_kl(q, source_mass) / max(pair.horizon, 1)
            loss_mass_slope = torch.sum(source_mass * growth_slope.square())
            loss_mass = (loss_mass_js
                         + MASS_KL_WEIGHT * loss_mass_kl
                         + MASS_SLOPE_WEIGHT * loss_mass_slope)
            mass_ess_ratio = 1.0 / (len(q) * torch.sum(q.square()))

            loss = (loss_flow + args.prior_weight * loss_prior
                    + args.mmd_weight * loss_mmd
                    + args.consistency_weight * loss_consistency
                    + loss_mass)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            losses["loss"].append(loss.item())
            losses["flow"].append(loss_flow.item())
            losses["prior"].append(loss_prior.item())
            losses["mmd"].append(loss_mmd.item())
            losses["consistency"].append(loss_consistency.item())
            losses["mass"].append(loss_mass.item())
            losses["mass_js"].append(loss_mass_js.item())
            losses["mass_kl"].append(loss_mass_kl.item())
            losses["mass_slope"].append(loss_mass_slope.item())
            losses["mass_ess_ratio"].append(mass_ess_ratio.item())
            losses["undercoverage"].append(float(np.mean(pair.undercoverage[p1])))

        row = {k: float(np.mean(v)) for k, v in losses.items()}
        row.update({
            "epoch": epoch + 1,
            "sigma": float(torch.exp(log_sigma1).item()),
            "horizon_counts": {
                str(h): int(np.sum(np.asarray(sampled_horizons) == h))
                for h in available_horizons
            },
        })
        history.append(row)
        print("UNIFIED", row, flush=True)
    return history


def exact_w2(true, pred):
    import ot as ot_lib
    weights_true = np.ones(len(true), np.float64) / len(true)
    weights_pred = np.ones(len(pred), np.float64) / len(pred)
    cost = np.ascontiguousarray(pairwise_distances(true, pred, metric="sqeuclidean", n_jobs=-1))
    return float(np.sqrt(ot_lib.emd2(weights_true, weights_pred, cost, numItermax=int(1e7))))


def load_checkpoint(path, model, prior, flow, embedding, log_sigma1, device):
    # torch.load uses pickle. Load only checkpoints from a trusted source.
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced
        checkpoint = torch.load(path, map_location=device)
    expected = {
        "codes": 1, "alpha": 0.5, "growth": True,
        "undercoverage_weight": 2.0, "mass_weight": 1.0,
        "mass_kl_weight": 0.1, "mass_slope_weight": 0.01,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Incompatible checkpoint {key}: {checkpoint.get(key)!r}; expected {value!r}")
    if checkpoint.get("gp_gate") is not None or checkpoint.get("no_gp_anchor") is False:
        raise ValueError("Checkpoint used an active GP anchor and is not compatible")
    if tuple(checkpoint["codebook_embed"].shape) != (1, 8):
        raise ValueError("Expected a single 8-dimensional code embedding")
    if tuple(checkpoint["log_sigma1"].shape) != (1,):
        raise ValueError("Expected one learned log-sigma")
    model.load_state_dict(checkpoint["ae"])
    prior.load_state_dict(checkpoint["prior"])
    flow.load_state_dict(checkpoint["flow"])
    with torch.no_grad():
        embedding.weight.copy_(checkpoint["codebook_embed"].to(device))
        log_sigma1.copy_(checkpoint["log_sigma1"].to(device))
    model.eval()
    prior.eval()
    flow.eval()
    embedding.eval()
    return checkpoint


def run(args):
    started = time.time()
    seed_all(args.seed)
    global ROOT, DATA
    ROOT = args.root.resolve()
    DATA = args.data_root.resolve() if args.data_root else ROOT / "new_data"
    original = CONFIGS[args.dataset]
    if args.dataset == "vitrobetacell":
        matrix = args.vitro_dir or DATA / "vitrobetacell/h5ad_x1"
        config = dataclass_replace(original, matrix=Path(matrix))
    else:
        matrix = DATA / original.matrix.relative_to(Path(__file__).resolve().parent / "new_data")
        metadata = (DATA / original.metadata.relative_to(Path(__file__).resolve().parent / "new_data")
                    if original.metadata else None)
        config = dataclass_replace(original, matrix=matrix, metadata=metadata)
    global OUT, CKPT, REPORT
    stem = f"{config.name}_protomass_clean{args.tag}"
    OUT = ROOT / f"{stem}_outputs"
    CKPT = ROOT / f"{stem}_checkpoints"
    REPORT = ROOT / f"{stem}_reports"
    for directory in (OUT, CKPT, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_tps, test_tps = config.train_tps, config.test_tps

    groups = load_data(config)
    model = AutoEncoder().to(device)
    prior = PriorNet().to(device)
    flow = UnifiedFlow().to(device)
    embedding = nn.Embedding(1, 8).to(device)
    nn.init.normal_(embedding.weight, mean=0.0, std=0.1)
    log_sigma1 = nn.Parameter(torch.zeros(1, device=device))

    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model, prior, flow,
                                     embedding, log_sigma1, device)
        if checkpoint.get("max_horizon", 3) != args.max_horizon:
            raise ValueError("--max-horizon must match the checkpoint")
        ae_history = flow_history = None
        pairs = []
        print("LOADED", args.checkpoint, flush=True)
    else:
        ae_history = train_ae(model, groups, train_tps, device, args.ae_epochs, args.seed)

    encoded = encode_groups(model, groups, device)
    if args.checkpoint:
        trained_horizons = list(range(1, min(args.max_horizon, len(train_tps) - 1) + 1))
    else:
        pairs = build_dynamics(encoded, train_tps, args.seed, args.max_horizon)
        trained_horizons = sorted({p.horizon for p in pairs})
        print("MULTI_HORIZON", {
            "trained_horizons": trained_horizons,
            "pair_counts": {h: sum(p.horizon == h for p in pairs) for h in trained_horizons},
        }, flush=True)
        flow_history = train_unified(prior, flow, embedding, log_sigma1, pairs,
                                     train_tps[-1], device, args.flow_epochs,
                                     args.seed + 200, args)
        model.eval()
        prior.eval()
        flow.eval()
        embedding.eval()
        torch.save({
            "ae": model.state_dict(), "prior": prior.state_dict(),
            "flow": flow.state_dict(),
            "codebook_centers": torch.zeros(1, 32),  # legacy format only
            "codebook_embed": embedding.weight.detach().cpu(),
            "log_sigma1": log_sigma1.detach().cpu(),
            "gp_gate": None, "alpha": 0.5, "codes": 1,
            "max_horizon": args.max_horizon,
            "horizon_weights": args.horizon_weights,
            "consistency_weight": args.consistency_weight,
            "ensemble_weights": args.ensemble_weights,
            "growth": True, "undercoverage_weight": 2.0,
            "mass_weight": 1.0, "mass_kl_weight": 0.1,
            "mass_slope_weight": 0.01,
            "no_gp_anchor": True,
        }, CKPT / f"{config.name}_medium_seed{args.seed}.pt")

    rng = np.random.default_rng(args.seed + 300)
    history = {}
    for source_tp in train_tps:
        if train_tps[-1] - source_tp < max(trained_horizons):
            pool = encoded[source_tp]
            history[source_tp] = pool[rng.choice(
                len(pool), args.n_generate, replace=len(pool) < args.n_generate
            )].copy()
    configured_ensemble = [float(x) for x in args.ensemble_weights.split(",")]

    def growth_distribution(source, source_tp, horizon):
        """Relative source contribution q and q/a for an equal-mass cell pool."""
        src_t = torch.from_numpy(source).to(device)
        with torch.no_grad():
            growth_time = source_tp / max(train_tps[-1], 1)
            log_rate, growth_slope = growth_log_rate(flow, src_t, growth_time)
            q_t = torch.softmax(horizon * log_rate, dim=0)
        q = q_t.detach().cpu().numpy().astype(np.float64)
        q /= q.sum()
        relative_multiplier = (len(source) * q).astype(np.float32)
        entropy = float(-np.sum(q * np.log(np.maximum(q, 1e-12))))
        ess = float(1.0 / np.sum(np.square(q)))
        return q, relative_multiplier, {
            "growth_log_rate_mean": float(log_rate.mean().item()),
            "growth_log_rate_std": float(log_rate.std().item()),
            "growth_slope_mean": float(growth_slope.mean().item()),
            "growth_slope_std": float(growth_slope.std().item()),
            "relative_multiplier_mean": float(relative_multiplier.mean()),
            "relative_multiplier_std": float(relative_multiplier.std()),
            "relative_multiplier_max": float(relative_multiplier.max()),
            "mass_weight_entropy": entropy,
            "mass_weight_ess": ess,
            "mass_weight_ess_ratio": ess / len(source),
        }

    def predict_direct(source, source_tp, horizon):
        """Generate one independent stochastic descendant per supplied ancestor."""
        biological_time = source_tp / train_tps[-1]
        src_t = torch.from_numpy(source).to(device)
        bt_t = torch.full((len(source), 1), float(biological_time), device=device)
        delta_t = torch.full((len(source), 1), horizon / train_tps[-1], device=device)
        emb = embedding.weight.expand(len(source), -1)
        sigma = torch.exp(log_sigma1).clamp(0.05, 5.0)
        with torch.no_grad():
            correction = prior(src_t, emb, bt_t, delta_t)
            eps = euler_sample(flow, src_t, emb, bt_t, delta_t,
                               steps=args.infer_steps)
            displacement = correction + sigma * eps
            candidate = np.clip(
                source + displacement.cpu().numpy(), -0.999, 0.999
            ).astype(np.float32)
        return candidate, {
            "source_tp": int(source_tp),
            "horizon": int(horizon),
            "weight": None,
            "displacement_norm_mean": float(
                torch.linalg.vector_norm(displacement, dim=1).mean().item()
            ),
        }

    metrics, diagnostics = {}, {}
    for target_tp in test_tps:
        candidates, candidate_mass, candidate_ancestors = [], [], []
        candidate_q, candidate_diagnostics, raw_weights = [], [], []
        for horizon in trained_horizons:
            source_tp = target_tp - horizon
            if source_tp not in history:
                continue
            weight = configured_ensemble[horizon - 1] if horizon <= len(configured_ensemble) else 0.0
            if weight <= 0:
                continue
            ancestor_pool = history[source_tp]
            q, relative_multiplier, growth_diag = growth_distribution(
                ancestor_pool, source_tp, horizon
            )
            ancestor_index = rng.choice(
                len(ancestor_pool), size=args.n_generate, replace=True, p=q
            )
            sampled_ancestors = ancestor_pool[ancestor_index]
            sampled_multiplier = relative_multiplier[ancestor_index]
            candidate, candidate_diag = predict_direct(
                sampled_ancestors, source_tp, horizon
            )
            candidate_diag.update(growth_diag)
            candidates.append(candidate)
            candidate_mass.append(sampled_multiplier)
            candidate_ancestors.append(ancestor_index)
            candidate_q.append(q)
            candidate_diagnostics.append(candidate_diag)
            raw_weights.append(weight)
        if not candidates:
            raise RuntimeError(f"No trained horizon can predict target timepoint {target_tp}")
        weights = np.asarray(raw_weights, dtype=np.float64)
        weights /= weights.sum()
        for weight, candidate_diag in zip(weights, candidate_diagnostics):
            candidate_diag["weight"] = float(weight)
        expert = rng.choice(len(candidates), size=args.n_generate, p=weights)
        source = np.empty_like(candidates[0])
        selected_multiplier = np.empty(args.n_generate, np.float32)
        selected_ancestor_index = np.empty(args.n_generate, np.int32)
        selected_source_tp = np.empty(args.n_generate, np.int16)
        selected_horizon = np.empty(args.n_generate, np.int8)
        for expert_index, (candidate, multiplier, ancestors, diag) in enumerate(
                zip(candidates, candidate_mass, candidate_ancestors,
                    candidate_diagnostics)):
            mask = expert == expert_index
            source[mask] = candidate[mask]
            selected_multiplier[mask] = multiplier[mask]
            selected_ancestor_index[mask] = ancestors[mask]
            selected_source_tp[mask] = diag["source_tp"]
            selected_horizon[mask] = diag["horizon"]
        effective_sample_size = float(1.0 / sum(
            np.sum(np.square(weight * q))
            for weight, q in zip(weights, candidate_q)
        ))
        history[target_tp] = source

        pred = decode(model, source, device)
        metrics[target_tp] = exact_w2(groups[target_tp], pred)
        diagnostics[target_tp] = {
            "sigma": float(torch.exp(log_sigma1).item()),
            "ensemble_mode": "mixture",
            "experts": candidate_diagnostics,
            "selected_mass_multiplier_mean": float(selected_multiplier.mean()),
            "selected_mass_multiplier_std": float(selected_multiplier.std()),
            "mass_weight_effective_pool_size": effective_sample_size,
            "mass_weight_effective_pool_ratio": float(
                effective_sample_size / (args.n_generate * len(candidates))
            ),
        }
        obs = pd.DataFrame({
            "source_tp": selected_source_tp,
            "horizon": selected_horizon,
            "growth_multiplier": selected_multiplier,
            "ancestor_index": selected_ancestor_index,
        })
        ad.AnnData(pred, obs=obs).write_h5ad(
            OUT / f"{config.name}_medium_t{target_tp}_seed{args.seed}.h5ad",
            compression="gzip"
        )
        print("EVAL", config.name, target_tp, metrics[target_tp], flush=True)

    report = {
        "dataset": config.name, "seed": args.seed,
        "train_tps": list(train_tps), "test_tps": list(test_tps),
        "n_generate": args.n_generate,
        "method": "DAFlow v2.4.2 (single-code, no-GP-anchor, nonstationary prototype mass)",
        "metric": "sqrt(POT emd2 with squared Euclidean cost), official LGP-OT evaluation",
        "w2": {str(k): v for k, v in metrics.items()},
        "diagnostics": {str(k): v for k, v in diagnostics.items()},
        "ae_final": ae_history[-1] if ae_history else None,
        "flow_final": flow_history[-1] if flow_history else None,
        "checkpoint_loaded": str(args.checkpoint) if args.checkpoint else None,
        "alpha": 0.5, "n_codes": 1,
        "mmd_weight": args.mmd_weight, "prior_weight": args.prior_weight,
        "max_horizon": args.max_horizon,
        "trained_horizons": trained_horizons,
        "horizon_weights": args.horizon_weights,
        "consistency_weight": args.consistency_weight,
        "consistency_every": args.consistency_every,
        "ensemble_weights": args.ensemble_weights,
        "ensemble_mode": "mixture",
        "growth": True,
        "undercoverage_weight": 2.0,
        "mass_weight": 1.0,
        "mass_kl_weight": 0.1,
        "mass_slope_weight": 0.01,
        "pair_mass_oracle": [{
            "source_t": int(pair.t),
            "target_t": int(pair.target_t),
            "oracle_mass_js": float(pair.oracle_mass_js),
        } for pair in pairs],
        "bt_jitter": args.bt_jitter,
        "paper_lgpot": {str(k): v for k, v in PAPER_LGPOT.get(config.name, {}).items()},
        "inference_transition": (
            "sample source ancestors from q, then generate independent stochastic "
            "descendants and mix direct-horizon experts"
        ),
        "runtime_seconds": time.time() - started,
    }
    path = REPORT / f"{config.name}_medium_seed{args.seed}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(CONFIGS), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="project root for reports, outputs, checkpoints, and gene cache")
    parser.add_argument("--data-root", type=Path,
                        help="dataset directory (default: ROOT/new_data)")
    parser.add_argument("--vitro-dir", type=Path,
                        help="vitrobetacell h5ad_x1 directory")
    parser.add_argument("--checkpoint", type=Path,
                        help="load a compatible trained checkpoint and run inference")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--tag", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ae-epochs", type=int, default=30)
    parser.add_argument("--flow-epochs", type=int, default=40)
    parser.add_argument("--n-generate", type=int, default=2000)
    parser.add_argument("--bt-jitter", type=float, default=2.0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--mmd-weight", type=float, default=0.05)
    parser.add_argument("--prior-weight", type=float, default=5.0)
    parser.add_argument("--mmd-steps", type=int, default=8)
    parser.add_argument("--infer-steps", type=int, default=24)
    parser.add_argument("--max-horizon", type=int, default=3)
    parser.add_argument("--horizon-weights", default="1,1,1")
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--consistency-every", type=int, default=2)
    parser.add_argument("--consistency-batch", type=int, default=256)
    parser.add_argument("--ensemble-weights", default="0.6,0.3,0.1")
    run(parser.parse_args())
