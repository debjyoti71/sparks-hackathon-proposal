"""Experimental: a small bidirectional GRU that runs beside the LightGBM models.

It reads the SAME per-frame features as the tree models (models.py), but as a sequence (every
SEQ_STEP-th frame, i.e. 6.25 per second), and predicts the same three things: P(dead), P(home | alive)
and P(direct turnover within ~0.5 s). Its probabilities are blended with the tree models' in
log-odds space with a weight chosen out-of-fold (0 = GRU ignored), see run_pipeline.train --gru.

Team symmetry is kept like everywhere else: each sequence is fed twice, with home and with away as
"own" team (labels swapped), and the two predictions are averaged.

Determinism: fixed seeds, torch.use_deterministic_algorithms(True), a fixed CPU thread count.
Requires PyTorch (CPU build is enough).
"""

import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from .labels import A, D, H, X

SEQ_STEP = 4                 # = STRIDE_TRAIN: training bundles are already every 4th frame
EPS = 1e-4


@dataclass
class GRUConfig:
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.1
    chunk: int = 256          # sequence length per training sample (256 x 4 frames ~ 41 s)
    batch: int = 32
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    threads: int = 4
    device: str = "cpu"        # "cuda" speeds up TRAINING only; inference after load() is always CPU


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def blend(p_tree, p_gru, w):
    """Log-odds blend; where the GRU has no prediction (outside play) the tree value is kept."""
    if p_gru is None or p_tree is None or not w:
        return p_tree
    out = _sigmoid((1 - w) * _logit(p_tree) + w * _logit(np.where(np.isnan(p_gru), 0.5, p_gru)))
    return np.where(np.isnan(p_gru), p_tree, out)


def _segments(period, step):
    """Row indices (into the bundle) of each half, every `step`-th row."""
    rows = np.arange(0, len(period), step)
    out = []
    for p in (1, 2):
        sel = rows[period[rows] == p]
        if len(sel) >= 16:
            out.append(sel)
    return out


def _build_net(n_in, cfg):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, cfg.hidden), nn.ReLU(), nn.Dropout(cfg.dropout))
            self.gru = nn.GRU(cfg.hidden, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                              bidirectional=True)
            self.out = nn.Linear(2 * cfg.hidden, 3)     # dead, own-team control, turnover

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            return self.out(h)

    return Net()


def _setup_torch(cfg):
    if cfg.device.startswith("cuda"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS
    import torch
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--gru-device cuda requested but torch.cuda.is_available() is False "
                           "(install a CUDA build of torch, see requirements-pipeline.txt)")
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)
    return torch


class GRUModel:
    def __init__(self, net=None, mu=None, sd=None, cfg=None, n_in=None):
        self.net, self.mu, self.sd, self.cfg, self.n_in = net, mu, sd, cfg or GRUConfig(), n_in

    # -- features --------------------------------------------------------------------------
    def _z(self, x):
        z = (x - self.mu) / self.sd
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -5, 5).astype(np.float32)

    # -- training --------------------------------------------------------------------------
    @classmethod
    def fit(cls, bundles, cfg=None, log=print):
        cfg = cfg or GRUConfig()
        torch = _setup_torch(cfg)
        rng = np.random.default_rng(cfg.seed)

        sample = np.vstack([np.hstack([b.Xc, b.Xo01])[b.codes != X] for b in bundles])
        with np.errstate(all="ignore"):
            mu = np.nanmean(sample, axis=0)
            sd = np.nanstd(sample, axis=0)
        mu = np.nan_to_num(mu).astype(np.float32)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)
        del sample
        model = cls(None, mu, sd, cfg, bundles[0].Xc.shape[1] + bundles[0].Xo01.shape[1])

        # (features, dead target, dead mask, own target, own mask, switch target) per half and orientation
        seqs = []
        for b in bundles:
            for rows in _segments(b.period, 1):                 # training bundles are already strided
                codes = b.codes[rows]
                graded = (codes != X).astype(np.float32)
                live = ((codes == H) | (codes == A)).astype(np.float32)
                sw = b.switch[rows].astype(np.float32) if b.switch is not None else np.zeros(len(rows), np.float32)
                for xo, own in ((b.Xo01, H), (b.Xo10, A)):
                    z = model._z(np.hstack([b.Xc[rows], xo[rows]]))
                    seqs.append((z, (codes == D).astype(np.float32), graded,
                                 (codes == own).astype(np.float32), live, sw))

        dev = torch.device(cfg.device)
        net = _build_net(model.n_in, cfg).to(dev)
        log("    GRU training on {}".format(dev))
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        L = cfg.chunk
        for epoch in range(cfg.epochs):
            items = []
            for si, s in enumerate(seqs):
                n = len(s[0])
                if n < L:
                    continue
                off = int(rng.integers(0, L))
                items += [(si, st) for st in range(off, n - L + 1, L)]
            order = rng.permutation(len(items))
            net.train()
            total, nb = 0.0, 0
            for k in range(0, len(order), cfg.batch):
                batch = [items[i] for i in order[k:k + cfg.batch]]
                arrs = [np.stack([seqs[si][j][st:st + L] for si, st in batch]) for j in range(6)]
                x, yd, md, yo, mo, ys = (torch.tensor(a, device=dev) for a in arrs)
                out = net(x)
                l_dead = (bce(out[..., 0], yd) * md).sum() / md.sum().clamp(min=1.0)
                l_own = (bce(out[..., 1], yo) * mo).sum() / mo.sum().clamp(min=1.0)
                l_sw = (bce(out[..., 2], ys) * md).sum() / md.sum().clamp(min=1.0)
                loss = l_dead + l_own + l_sw
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += float(loss.detach().cpu())
                nb += 1
            log("    GRU epoch {}/{}: loss {:.4f}".format(epoch + 1, cfg.epochs, total / max(nb, 1)))
        net.eval()
        model.net = net
        return model

    # -- inference -------------------------------------------------------------------------
    def predict(self, b, step=SEQ_STEP):
        """-> (p_dead, p_home, p_switch) aligned with the bundle's rows; NaN outside play.
        Predicted every `step`-th row and linearly interpolated (in log-odds) in between."""
        torch = _setup_torch(self.cfg)
        dev = torch.device(self.cfg.device)
        T = len(b.period)
        out = {k: np.full(T, np.nan) for k in ("dead", "home", "switch")}
        for rows in _segments(b.period, step):
            logits = []
            for xo in (b.Xo01, b.Xo10):
                z = self._z(np.hstack([b.Xc[rows], xo[rows]]))
                with torch.no_grad():
                    logits.append(self.net(torch.tensor(z, device=dev)[None])[0].cpu().numpy().astype(np.float64))
            full = np.arange(rows[0], rows[-1] + 1)
            d = 0.5 * (_sigmoid(logits[0][:, 0]) + _sigmoid(logits[1][:, 0]))
            h = 0.5 * (_sigmoid(logits[0][:, 1]) + 1.0 - _sigmoid(logits[1][:, 1]))
            s = 0.5 * (_sigmoid(logits[0][:, 2]) + _sigmoid(logits[1][:, 2]))
            for key, val in (("dead", d), ("home", h), ("switch", s)):
                out[key][full] = _sigmoid(np.interp(full, rows, _logit(val)))
        return out["dead"], out["home"], out["switch"]

    # -- persistence -----------------------------------------------------------------------
    def save(self, folder):
        import torch
        os.makedirs(folder, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.net.state_dict().items()}, os.path.join(folder, "gru.pt"))
        np.savez(os.path.join(folder, "gru_norm.npz"), mu=self.mu, sd=self.sd)
        with open(os.path.join(folder, "gru_cfg.json"), "w") as handle:
            json.dump({"n_in": int(self.n_in), **asdict(self.cfg)}, handle, indent=2)

    @classmethod
    def load(cls, folder):
        import torch
        with open(os.path.join(folder, "gru_cfg.json")) as handle:
            raw = json.load(handle)
        n_in = raw.pop("n_in")
        raw["device"] = "cpu"        # the submission is always produced on CPU, on any machine
        cfg = GRUConfig(**raw)
        _setup_torch(cfg)
        net = _build_net(n_in, cfg)
        net.load_state_dict(torch.load(os.path.join(folder, "gru.pt"), map_location="cpu"))
        net.eval()
        z = np.load(os.path.join(folder, "gru_norm.npz"))
        return cls(net, z["mu"], z["sd"], cfg, n_in)
