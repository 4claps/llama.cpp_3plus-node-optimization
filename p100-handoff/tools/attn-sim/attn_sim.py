#!/usr/bin/env python3
"""Score attention approximations offline against exact attention, on real Q/K/V dumped by
GGML_CUDA_FA_DUMP (fattn-gemm.cu). For sampled queries it reports, per method, the relative L2
error of the attention output vs exact (float64) and the softmax mass the method keeps exactly.

usage: attn_sim.py <dump dir> <call prefix, e.g. c000_d0> [--queries-per-head 32] [--heads 0,5]
"""
import argparse, os, sys, time
import numpy as np


def load_meta(path):
    t = open(path).read().split()
    return {t[i]: float(t[i + 1]) if t[i] == "scale" else int(t[i + 1]) for i in range(0, len(t), 2)}


def deq_q40(raw, nrows, D):
    """raw q4_0 rows -> float32 [nrows, D]"""
    nb = D // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(nrows, nb, 18)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32)          # [nrows, nb, 1]
    qs = b[:, :, 2:18]
    lo = (qs & 0x0F).astype(np.float32) - 8.0
    hi = (qs >> 4).astype(np.float32) - 8.0
    x = np.concatenate([lo, hi], axis=2) * d                             # [nrows, nb, 32]
    return x.reshape(nrows, D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("prefix")
    ap.add_argument("--queries-per-head", type=int, default=32)
    ap.add_argument("--heads", default="")
    ap.add_argument("--topk", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--ranks", default="16,32,64")
    a = ap.parse_args()

    m = load_meta(os.path.join(a.dir, a.prefix + "_meta.txt"))
    D, DV, nt, nh, nhkv, nkv, scale = m["D"], m["DV"], m["nt"], m["nh"], m["nhkv"], m["nkv"], m["scale"]
    gqa = nh // nhkv
    Q = np.fromfile(os.path.join(a.dir, a.prefix + "_q.bin"), dtype=np.float32).reshape(nh, nt, D)
    rb = D // 32 * 18
    kraw = open(os.path.join(a.dir, a.prefix + "_k.bin"), "rb").read()
    vraw = open(os.path.join(a.dir, a.prefix + "_v.bin"), "rb").read()
    mf = np.fromfile(os.path.join(a.dir, a.prefix + "_maskfirst.bin"), dtype=np.float32)
    heads = [int(x) for x in a.heads.split(",")] if a.heads else list(range(nh))
    topks = [int(x) for x in a.topk.split(",")]
    ranks = [int(x) for x in a.ranks.split(",")]
    print(f"{a.prefix}: D {D} nt {nt} nh {nh} nhkv {nhkv} nkv {nkv} scale {scale:.4g}", flush=True)

    rng = np.random.default_rng(0)
    res = {}   # method -> list of (rel err, mass kept)
    spectra = {}
    cache = {}
    for h in heads:
        kvh = h // gqa
        if kvh not in cache:
            K = deq_q40(kraw[kvh*nkv*rb:(kvh + 1)*nkv*rb], nkv, D)
            V = deq_q40(vraw[kvh*nkv*rb:(kvh + 1)*nkv*rb], nkv, DV)
            L = int(mf.min())   # keys every sampled query may see (causal prefix); per-query bound below
            # PCA of keys and values (uncentered: scoring is a plain dot product)
            ck = (K[:L].astype(np.float64).T @ K[:L].astype(np.float64)) / L
            ek, Uk = np.linalg.eigh(ck); Uk = Uk[:, ::-1]; ek = ek[::-1]
            cv = (V[:L].astype(np.float64).T @ V[:L].astype(np.float64)) / L
            ev, Uv = np.linalg.eigh(cv); Uv = Uv[:, ::-1]; ev = ev[::-1]
            spectra[kvh] = (np.cumsum(ek) / ek.sum(), np.cumsum(ev) / ev.sum())
            cache = {kvh: (K, V, Uk, Uv)}
        K, V, Uk, Uv = cache[kvh]
        ts = np.sort(rng.choice(nt, size=min(a.queries_per_head, nt), replace=False))
        for t in ts:
            n = int(mf[t])                      # keys [0, n) are unmasked for this query
            q = Q[h, t].astype(np.float64)
            Kn, Vn = K[:n].astype(np.float64), V[:n].astype(np.float64)
            s = scale * (Kn @ q)
            w = np.exp(s - s.max()); Z = w.sum(); p = w / Z
            o = p @ Vn
            on = np.linalg.norm(o)
            # yardstick: the shipped kernel's rounding (fp16 Q and K, logits in fp32, P rounded to fp16)
            s16 = scale * (Kn.astype(np.float16).astype(np.float64) @ q.astype(np.float16).astype(np.float64))
            p16 = np.exp(s16 - s16.max()).astype(np.float16).astype(np.float64)
            o16 = (p16 @ Vn) / p16.sum()
            res.setdefault("yardstick: shipped fp16 rounding", []).append((np.linalg.norm(o16 - o)/on, 1.0))
            order = np.argsort(-s)
            for k in topks:
                if k >= n: continue
                sel = order[:k]
                ok = (w[sel] @ Vn[sel]) / w[sel].sum()
                res.setdefault(f"exact top-{k}", []).append((np.linalg.norm(ok - o)/on, p[sel].sum()))
            for r in ranks:
                Ur = Uk[:, :r]
                sr = scale * ((Kn @ Ur) @ (Ur.T @ q))   # low-rank scores
                orr = np.argsort(-sr)
                for k in topks:
                    if k >= n: continue
                    sel = orr[:k]
                    ok = (w[sel] @ Vn[sel]) / w[sel].sum()
                    res.setdefault(f"rank-{r} select, top-{k}", []).append((np.linalg.norm(ok - o)/on, p[sel].sum()))
                    # tail approximated: low-rank scores for the unselected keys, V projected to rank 2r
                    mask = np.ones(n, bool); mask[sel] = False
                    Uv2 = Uv[:, :2*r]
                    wt = np.exp(sr[mask] - s.max())          # tail weights from approximate scores
                    vt = (wt @ (Vn[mask] @ Uv2)) @ Uv2.T
                    num = w[sel] @ Vn[sel] + vt
                    den = w[sel].sum() + wt.sum()
                    ok2 = num / den
                    res.setdefault(f"rank-{r} select, top-{k} + rank-{r} tail", []).append((np.linalg.norm(ok2 - o)/on, 1.0))
        print(f"head {h} done", flush=True)

    for kvh, (sk, sv) in spectra.items():
        print(f"kv head {kvh} PCA variance captured, K at r=16/32/64/128: "
              f"{sk[15]:.3f} {sk[31]:.3f} {sk[63]:.3f} {sk[127]:.3f} | V: {sv[15]:.3f} {sv[31]:.3f} {sv[63]:.3f} {sv[127]:.3f}")
    print(f"{'method':45s} {'rel err mean':>12s} {'p50':>9s} {'p99':>9s} {'mass kept':>9s}")
    for name, v in res.items():
        e = np.array([x[0] for x in v]); mk = np.array([x[1] for x in v])
        print(f"{name:45s} {e.mean():12.5f} {np.percentile(e,50):9.5f} {np.percentile(e,99):9.5f} {mk.mean():9.4f}")


if __name__ == "__main__":
    main()
