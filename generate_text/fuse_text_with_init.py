"""
Fuse text embeddings (.pt) with original features in an NPZ (x0/x1/...)
- Match by node count (N). Only layers with the same number of rows will be fused.
- Optional L2 normalization for both sides, with alpha/beta scaling before concat.
- Save to a new NPZ, keeping all original keys and adding x{layer}_fused.
"""

import os
import os.path as osp
import re
import glob
import argparse
import numpy as np
import torch


def extract_layers(npz_obj):
    """Return sorted list of layer indices from keys like 'x0', 'x1', ..."""
    layers = []
    for k in npz_obj.files:
        if re.fullmatch(r"x\d+", k):
            # ensure it's a 2D array
            v = npz_obj[k]
            if getattr(v, "ndim", 0) == 2:
                idx = int(k[1:])
                layers.append(idx)
    return sorted(layers)


def l2norm(x: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def load_text_embedding(pt_path: str) -> np.ndarray:
    if not osp.exists(pt_path):
        raise FileNotFoundError(pt_path)
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().numpy()
    else:
        arr = np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError(f"Text embedding must be 2D, got shape {arr.shape} from {pt_path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Found NaN/Inf in text embedding: {pt_path}")
    return arr.astype(np.float32, copy=False)


def fuse_one_layer(X: np.ndarray, T: np.ndarray,
                   normalize_x: bool, normalize_t: bool,
                   alpha: float, beta: float) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    T = T.astype(np.float32, copy=False)
    if normalize_x:
        X = l2norm(X)
    if normalize_t:
        T = l2norm(T)
    return np.concatenate([alpha * X, beta * T], axis=1)


def main():
    p = argparse.ArgumentParser(description="Fuse text embeddings (.pt) with NPZ xN features.")
    p.add_argument("--npz_path", default=data/douban.npz, help="Path to original .npz (contains x0/x1/...)")
    # 模式一：单个 .pt
    p.add_argument("--text_pt", default=None, help="Path to a single .pt text embedding")
    # 模式二：目录内多个 .pt
    p.add_argument("--text_dir", default=None, help="Directory containing multiple .pt files")
    p.add_argument("--pt_glob", default="*.pt", help="Glob to select .pt files in text_dir (default: *.pt)")

    p.add_argument("--normalize_x", action="store_true", help="L2-normalize original x before concat")
    p.add_argument("--normalize_t", action="store_true", help="L2-normalize text emb before concat")
    p.add_argument("--alpha", type=float, default=1.0, help="Scale for original x before concat")
    p.add_argument("--beta", type=float, default=1.0, help="Scale for text emb before concat")

    p.add_argument("--out_path", default=None, help="Output fused npz path (default: {stem}_fused.npz)")
    args = p.parse_args()

    if not osp.exists(args.npz_path):
        raise FileNotFoundError(args.npz_path)

    if not args.text_pt and not args.text_dir:
        raise ValueError("You must provide either --text_pt (single file) or --text_dir (directory of .pt files).")

    data = np.load(args.npz_path, allow_pickle=True)
    layers = extract_layers(data)
    if not layers:
        raise ValueError("No xN layers (2D arrays) found in the NPZ.")

    # Prepare outputs: keep all original keys
    out_dict = {k: data[k] for k in data.files}

    # Build a list of text embeddings to try
    text_sources = []
    if args.text_pt:
        text_sources.append(args.text_pt)
    if args.text_dir:
        text_sources.extend(sorted(glob.glob(osp.join(args.text_dir, args.pt_glob))))

    # Preload all text emb files and index by (N, D)
    text_emb_by_rows = {}  # key: N -> list of (path, array)
    for pt in text_sources:
        T = load_text_embedding(pt)
        N, D = T.shape
        text_emb_by_rows.setdefault(N, []).append((pt, T))

    fused_count = 0
    for l in layers:
        X = data[f"x{l}"]
        N_x = X.shape[0]

        # find a T whose N matches
        candidates = text_emb_by_rows.get(N_x, [])
        if not candidates:
            # no matching text embedding for this layer; skip
            continue

        # choose the first candidate by default
        pt_path, T = candidates[0]
        XF = fuse_one_layer(X, T, args.normalize_x, args.normalize_t, args.alpha, args.beta)
        key = f"x{l}_fused"
        out_dict[key] = XF
        fused_count += 1
        print(f"[layer x{l}] fused -> {key}: {X.shape[1]} + {T.shape[1]} = {XF.shape[1]} dims (pt: {osp.basename(pt_path)})")

    if fused_count == 0:
        raise RuntimeError("No layers were fused. (No text embedding with matching row count found.)")

    stem = osp.splitext(osp.basename(args.npz_path))[0]
    out_path = args.out_path or osp.join(osp.dirname(args.npz_path), f"{stem}_fused.npz")
    np.savez_compressed(out_path, **out_dict)
    print("Saved fused npz ->", out_path)


if __name__ == "__main__":
    main()
