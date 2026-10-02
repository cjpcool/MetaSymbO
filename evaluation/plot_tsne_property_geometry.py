"""
Plot property and geometry distributions with t-SNE for rebuttal analysis.

This script compares the MetaModulus training split with one or more generated
result folders, e.g., MetaSymbO-Mix and MetaSymbO-Union.

Example:
    python plot_tsne_property_geometry.py \
        --dataset_root /home/grads/jianpengc/datasets/metamaterial/LatticeModulus \
        --dataset_type modulus \
        --file_name data \
        --tsne_iter 250 \
        --gen_paths METASYMBO-Mix=/home/grads/jianpengc/projects/materials/MetaSymbO/results/completely_mix \
        --out_dir ./tsne_rebuttal_mix \
        --max_train 100 \
        --max_gen 100

Outputs:
    tsne_geometry.png / tsne_geometry.pdf / tsne_geometry_embedding.npz
    tsne_property.png / tsne_property.pdf / tsne_property_embedding.npz
"""

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Make project imports work when this script is placed in an eval/scripts folder.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR)
sys.path.append(os.path.join(_THIS_DIR, ".."))
sys.path.append(os.path.join(_THIS_DIR, "../.."))

from datasets.dataset_truss import LatticeModulus, LatticeStiffness  # noqa: E402
from eval_utils import lattice_params_to_matrix, frac_to_cart_coords, cart_to_frac_coords  # noqa: E402


MAX_NODE_NUM = 100


def to_numpy(x):
    """Convert torch/numpy/list/scalar object to numpy array or None."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == object and arr.shape == ():
        item = arr.item()
        if item is None:
            return None
        return to_numpy(item)
    return arr


def npz_get_optional(npz, key: str, default=None):
    if key not in npz.files:
        return default
    return to_numpy(npz[key])


def lattice_matrix_from_npz(npz) -> np.ndarray:
    """Load lattice matrix from `vector`, or rebuild it from lengths/angles."""
    vector = npz_get_optional(npz, "vector")
    if vector is not None:
        return np.asarray(vector, dtype=float).reshape(3, 3)

    lengths = npz_get_optional(npz, "lengths")
    angles = npz_get_optional(npz, "angles")
    if lengths is None or angles is None:
        raise KeyError("Generated npz must contain either `vector` or `lengths` + `angles`.")
    lengths = np.asarray(lengths, dtype=float).reshape(-1)
    angles = np.asarray(angles, dtype=float).reshape(-1)
    return lattice_params_to_matrix(
        lengths[0], lengths[1], lengths[2], angles[0], angles[1], angles[2]
    )


def read_generated_npz(path: str, prop_key: str = "prop_list") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read one generated structure file.

    Returns:
        cart_coords: (N, 3) Cartesian coordinates.
        lattice_vector: (3, 3) lattice matrix.
        prop: optional property vector from `prop_key`.
    """
    npz = np.load(path, allow_pickle=True)
    lattice_vector = lattice_matrix_from_npz(npz)

    frac_coords = npz_get_optional(npz, "frac_coords")
    cart_coords = npz_get_optional(npz, "cart_coords")
    if frac_coords is None and cart_coords is None:
        raise KeyError(f"{path} must contain either `frac_coords` or `cart_coords`.")

    if cart_coords is None:
        frac_coords = np.asarray(frac_coords, dtype=float)
        cart_coords = frac_to_cart_coords(frac_coords, lattice_vector, len(frac_coords))
    else:
        cart_coords = np.asarray(cart_coords, dtype=float)

    if frac_coords is None:
        # Not needed for t-SNE, but this catches malformed lattice/coords early.
        _ = cart_to_frac_coords(cart_coords, lattice_vector, len(cart_coords))

    prop = npz_get_optional(npz, prop_key)
    if prop is not None:
        prop = np.asarray(prop, dtype=float).reshape(-1)
    return cart_coords, lattice_vector, prop


def list_npz_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if len(files) == 0:
        files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found under: {path}")
    return files


def load_generated_set(path: str, prop_key: str = "prop_list") -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
    files = list_npz_files(path)
    coords, lattices, props = [], [], []
    for file_path in tqdm(files, desc=f"Loading generated: {os.path.basename(path.rstrip('/')) or path}"):
        try:
            c, l, p = read_generated_npz(file_path, prop_key=prop_key)
            coords.append(c)
            lattices.append(l)
            props.append(p)
        except Exception as exc:
            print(f"[WARN] Skip invalid generated file: {file_path}. Reason: {exc}")
    if len(coords) == 0:
        raise ValueError(f"No valid generated structures were loaded from: {path}")
    return coords, lattices, props


def load_metamodulus_dataset(dataset_root: str, dataset_type: str, file_name: str):
    if dataset_type.lower() in ["modulus", "latticemodulus"]:
        return LatticeModulus(dataset_root, file_name=file_name)
    if dataset_type.lower() in ["stiffness", "latticestiffness"]:
        return LatticeStiffness(dataset_root, file_name=file_name)
    raise ValueError("`dataset_type` must be either `modulus` or `stiffness`.")


def filter_dataset_by_size(dataset, max_node_num: int = MAX_NODE_NUM):
    indices = []
    for i, data in enumerate(tqdm(dataset, desc="Filtering dataset by graph size")):
        num_atoms = int(data.num_atoms) if not isinstance(data.num_atoms, torch.Tensor) else int(data.num_atoms.item())
        num_edges = int(data.num_edges) if not isinstance(data.num_edges, torch.Tensor) else int(data.num_edges.item())
        if num_atoms <= max_node_num and num_edges <= max_node_num * 2:
            indices.append(i)
    return dataset[indices]


def get_train_split(dataset, train_size: int, seed: int):
    split_dict = dataset.get_idx_split(len(dataset), train_size, 1, seed=seed)
    train_idx = split_dict["train"].tolist()
    return dataset[train_idx]


def extract_dataset_arrays(dataset) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    coords, lattices, props = [], [], []
    for data in tqdm(dataset, desc="Extracting train arrays"):
        coords.append(to_numpy(data.cart_coords).astype(float))
        if hasattr(data, "vector"):
            lattices.append(to_numpy(data.vector).astype(float).reshape(3, 3))
        elif hasattr(data, "lengths") and hasattr(data, "angles"):
            lengths = to_numpy(data.lengths).reshape(-1)
            angles = to_numpy(data.angles).reshape(-1)
            lattices.append(lattice_params_to_matrix(*lengths[:3], *angles[:3]))
        else:
            raise AttributeError("Dataset item must contain `vector` or `lengths` + `angles`.")

        if not hasattr(data, "y"):
            raise AttributeError("Dataset item must contain property tensor `y` for property t-SNE.")
        props.append(to_numpy(data.y).reshape(-1).astype(float))
    return coords, lattices, np.stack(props, axis=0)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = np.linalg.norm(u) * np.linalg.norm(v) + 1e-12
    cos_val = float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_val)))


def lattice_lengths_angles(lattice: np.ndarray) -> np.ndarray:
    a, b, c = np.asarray(lattice, dtype=float).reshape(3, 3)
    lengths = np.array([np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)], dtype=float)
    # Conventional angle order: alpha=<b,c>, beta=<a,c>, gamma=<a,b>.
    angles = np.array([angle_between(b, c), angle_between(a, c), angle_between(a, b)], dtype=float)
    return np.concatenate([lengths, angles], axis=0)


def pad_or_truncate(values: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] >= length:
        return values[:length]
    out = np.full(length, pad_value, dtype=float)
    out[: values.shape[0]] = values
    return out


def geometry_fingerprint(
    cart_coords: np.ndarray,
    lattice: np.ndarray,
    max_nodes: int = 100,
    max_pair_dists: int = 1024,
) -> np.ndarray:
    """Build a fixed-length geometry descriptor for t-SNE.

    This descriptor is intentionally simple and deterministic:
      1) number of nodes;
      2) lattice lengths and angles;
      3) sorted node radial distances from cell centroid;
      4) sorted pairwise node distances.

    It supports variable-size generated graphs and includes lattice scale/shape.
    """
    coords = np.asarray(cart_coords, dtype=float)
    lattice_feat = lattice_lengths_angles(lattice)
    n = coords.shape[0]

    centered = coords - coords.mean(axis=0, keepdims=True)
    radial = np.sort(np.linalg.norm(centered, axis=1))
    radial = pad_or_truncate(radial, max_nodes)

    if n >= 2:
        diff = coords[:, None, :] - coords[None, :, :]
        dist_mat = np.sqrt(np.sum(diff * diff, axis=-1))
        pair_idx = np.triu_indices(n, k=1)
        pair_dists = np.sort(dist_mat[pair_idx])
    else:
        pair_dists = np.array([], dtype=float)
    pair_dists = pad_or_truncate(pair_dists, max_pair_dists)

    return np.concatenate([
        np.array([float(n)], dtype=float),
        lattice_feat,
        radial,
        pair_dists,
    ])


def central_symmetry_score(coords: np.ndarray, error_bar: float = 0.1) -> float:
    coords = np.asarray(coords, dtype=float)
    center = np.array([
        (coords[:, 0].max() + coords[:, 0].min()) / 2,
        (coords[:, 1].max() + coords[:, 1].min()) / 2,
        (coords[:, 2].max() + coords[:, 2].min()) / 2,
    ])
    dist = coords - center
    new_dist = dist[:, None, :] + dist[None, :, :]
    central_dist = np.sqrt(np.sum(new_dist * new_dist, axis=-1))
    is_symmetry_per_node = (np.isclose(central_dist, 0, atol=error_bar)).sum(axis=-1) > 0
    symmetry_node_num = is_symmetry_per_node.sum() + 1e-6
    symmetry_node_rate = is_symmetry_per_node.sum() / coords.shape[0]
    max_error = max(np.sqrt(np.sum(dist * dist, axis=-1)).max(), 1e-12)
    s_error_i = central_dist.min(axis=-1)
    s_error_i_ratio = (max_error - s_error_i) / max_error
    return float(symmetry_node_rate * (1.0 / symmetry_node_num) * (is_symmetry_per_node * s_error_i_ratio).sum())


def is_periodic_necessary_condition(coords: np.ndarray, lattice_vector: np.ndarray, error_bar: float = 0.1) -> bool:
    coords = np.asarray(coords, dtype=float)
    lattice_vector = np.asarray(lattice_vector, dtype=float).reshape(3, 3)
    for d in range(3):
        find_period = False
        for i in range(len(coords)):
            shifted = coords[i] + lattice_vector[d]
            dist = np.abs(shifted - coords)
            if np.any(np.isclose(dist.sum(-1), 0, atol=error_bar)):
                find_period = True
                break
        if not find_period:
            return False
    return True


def filter_valid_generated(
    coords: List[np.ndarray],
    lattices: List[np.ndarray],
    props: List[Optional[np.ndarray]],
    periodic_error_bar: float,
    symmetry_error_bar: float,
    symmetry_threshold: float,
):
    kept_c, kept_l, kept_p = [], [], []
    skipped = 0
    for c, l, p in tqdm(list(zip(coords, lattices, props)), desc="Filtering valid generated structures"):
        periodic = is_periodic_necessary_condition(c, l, error_bar=periodic_error_bar)
        symmetry = central_symmetry_score(c, error_bar=symmetry_error_bar) > symmetry_threshold
        if periodic and symmetry:
            kept_c.append(c)
            kept_l.append(l)
            kept_p.append(p)
        else:
            skipped += 1
    print(f"Valid-filter kept {len(kept_c)} / {len(coords)} generated structures; skipped {skipped}.")
    if len(kept_c) == 0:
        raise ValueError("No generated structures remain after validity filtering.")
    return kept_c, kept_l, kept_p


def sample_group(
    coords: List[np.ndarray],
    lattices: List[np.ndarray],
    props: Optional[np.ndarray],
    max_size: Optional[int],
    seed: int,
):
    n = len(coords)
    if max_size is None or max_size <= 0 or max_size >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=max_size, replace=False))
    coords_s = [coords[i] for i in idx]
    lattices_s = [lattices[i] for i in idx]
    props_s = None if props is None else np.asarray(props)[idx]
    return coords_s, lattices_s, props_s


def stack_available_props(props: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = [p for p in props if p is not None]
    if len(valid) == 0:
        return None
    dim = valid[0].shape[0]
    valid = [p for p in valid if p.shape[0] == dim]
    if len(valid) == 0:
        return None
    return np.stack(valid, axis=0)


def parse_gen_paths(items: List[str]) -> Dict[str, str]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                "Each --gen_paths entry must use NAME=/path format, e.g., METASYMBO-Mix=/Path/results/mix"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if len(name) == 0 or len(path) == 0:
            raise ValueError(f"Invalid --gen_paths entry: {item}")
        out[name] = path
    return out


def make_tsne(features_by_group: Dict[str, np.ndarray], seed: int, perplexity: float, max_iter: int):
    labels = []
    X_parts = []
    for name, X in features_by_group.items():
        X = np.asarray(X, dtype=float)
        X_parts.append(X)
        labels.extend([name] * X.shape[0])
    X_all = np.concatenate(X_parts, axis=0)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    X_all = StandardScaler().fit_transform(X_all)

    n = X_all.shape[0]
    if n <= 3:
        raise ValueError("Need more than 3 total samples for t-SNE.")
    perplexity = min(float(perplexity), max(2.0, float((n - 1) // 3)))

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    try:
        tsne = TSNE(**kwargs, max_iter=max_iter)
    except TypeError:
        tsne = TSNE(**kwargs, n_iter=max_iter)
    emb = tsne.fit_transform(X_all)
    return emb, np.asarray(labels)


def save_tsne_plot(
    emb: np.ndarray,
    labels: np.ndarray,
    out_prefix: str,
    title: str,
    point_size: float = 20.0,
    alpha: float = 0.72,
):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    names = list(dict.fromkeys(labels.tolist()).keys())
    palette = plt.get_cmap("Set2").colors
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=260)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#f6f8fb")

    for idx, name in enumerate(names):
        mask = labels == name
        color = palette[idx % len(palette)]
        marker = marker_cycle[idx % len(marker_cycle)]
        ax.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=max(point_size, 12.0) * 1.15,
            alpha=min(alpha + 0.08, 0.92),
            c=[color],
            marker=marker,
            edgecolors="white",
            linewidths=0.45,
            label=f"{name}",
            rasterized=True,
        )

    ax.set_title(title, fontsize=14, pad=12, weight="semibold")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
    ax.tick_params(axis="both", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#8f98a3")
        ax.spines[spine].set_linewidth(0.8)

    legend = ax.legend(
        frameon=True,
        fontsize=9,
        loc="best",
        fancybox=True,
        framealpha=0.92,
        borderpad=0.6,
        handletextpad=0.5,
        labelspacing=0.45,
    )
    legend.get_frame().set_edgecolor("#d6dce5")
    legend.get_frame().set_linewidth(0.8)

    fig.tight_layout(pad=1.2)
    fig.savefig(out_prefix + ".png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(out_prefix + ".pdf", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    np.savez(out_prefix + "_embedding.npz", embedding=emb, labels=labels)
    print(f"Saved: {out_prefix}.png")
    print(f"Saved: {out_prefix}.pdf")
    print(f"Saved: {out_prefix}_embedding.npz")


def align_property_dims(features_by_group: Dict[str, np.ndarray], truncate: bool = True) -> Dict[str, np.ndarray]:
    dims = {name: np.asarray(X).shape[1] for name, X in features_by_group.items()}
    if len(set(dims.values())) == 1:
        return features_by_group
    if not truncate:
        raise ValueError(f"Property dimensions are inconsistent across groups: {dims}")
    min_dim = min(dims.values())
    print(f"[WARN] Property dimensions differ: {dims}. Truncating all groups to first {min_dim} dims.")
    return {name: np.asarray(X)[:, :min_dim] for name, X in features_by_group.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="modulus", choices=["modulus", "stiffness"])
    parser.add_argument("--file_name", type=str, default="data")
    parser.add_argument("--train_size", type=int, default=8000)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--gen_paths", nargs="+", required=True, help="Entries in NAME=/path format.")
    parser.add_argument("--gen_prop_key", type=str, default="prop_list")
    parser.add_argument("--out_dir", type=str, default="./tsne_rebuttal")
    parser.add_argument("--max_train", type=int, default=2000)
    parser.add_argument("--max_gen", type=int, default=2000)
    parser.add_argument("--max_nodes", type=int, default=100)
    parser.add_argument("--max_pair_dists", type=int, default=1024)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--tsne_iter", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_only", action="store_true", help="Only plot generated structures passing periodicity and central symmetry checks.")
    parser.add_argument("--periodic_error_bar", type=float, default=0.1)
    parser.add_argument("--symmetry_error_bar", type=float, default=0.1)
    parser.add_argument("--symmetry_threshold", type=float, default=0.0)
    parser.add_argument("--skip_property", action="store_true")
    parser.add_argument("--skip_geometry", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_metamodulus_dataset(args.dataset_root, args.dataset_type, args.file_name)
    dataset = filter_dataset_by_size(dataset, max_node_num=MAX_NODE_NUM)
    train_data = get_train_split(dataset, train_size=args.train_size, seed=args.split_seed)
    train_coords, train_lattices, train_props = extract_dataset_arrays(train_data)
    train_coords, train_lattices, train_props = sample_group(
        train_coords, train_lattices, train_props, args.max_train, seed=args.seed
    )

    coords_by_group = {"Train": train_coords}
    lattices_by_group = {"Train": train_lattices}
    props_by_group = {"Train": train_props}

    gen_paths = parse_gen_paths(args.gen_paths)
    for method_name, method_path in gen_paths.items():
        gen_coords, gen_lattices, gen_props_list = load_generated_set(method_path, prop_key=args.gen_prop_key)
        if args.valid_only:
            gen_coords, gen_lattices, gen_props_list = filter_valid_generated(
                gen_coords,
                gen_lattices,
                gen_props_list,
                periodic_error_bar=args.periodic_error_bar,
                symmetry_error_bar=args.symmetry_error_bar,
                symmetry_threshold=args.symmetry_threshold,
            )

        gen_props = stack_available_props(gen_props_list)
        # If some files have no property, geometry can still be plotted. For property
        # t-SNE, this method will be skipped with a warning.
        gen_coords, gen_lattices, _ = sample_group(
            gen_coords, gen_lattices, None, args.max_gen, seed=args.seed
        )
        if gen_props is not None:
            # Keep property sampling aligned only when every loaded file had property.
            # Otherwise property t-SNE still uses available property rows independently.
            _, _, gen_props = sample_group(
                list(range(len(gen_props))), list(range(len(gen_props))), gen_props, args.max_gen, seed=args.seed
            )

        coords_by_group[method_name] = gen_coords
        lattices_by_group[method_name] = gen_lattices
        if gen_props is not None:
            props_by_group[method_name] = gen_props
        else:
            print(f"[WARN] No `{args.gen_prop_key}` property vectors found for {method_name}; skip it in property t-SNE.")

    if not args.skip_geometry:
        geom_features_by_group = {}
        for name in coords_by_group.keys():
            feats = []
            for c, l in tqdm(list(zip(coords_by_group[name], lattices_by_group[name])), desc=f"Geometry features: {name}"):
                feats.append(
                    geometry_fingerprint(c, l, max_nodes=args.max_nodes, max_pair_dists=args.max_pair_dists)
                )
            geom_features_by_group[name] = np.stack(feats, axis=0)

        geom_emb, geom_labels = make_tsne(
            geom_features_by_group, seed=args.seed, perplexity=args.perplexity, max_iter=args.tsne_iter
        )
        save_tsne_plot(
            geom_emb,
            geom_labels,
            os.path.join(args.out_dir, "tsne_geometry"),
            title="Geometry distribution: train vs. generated",
        )

    if not args.skip_property:
        if len(props_by_group) < 2:
            print("[WARN] Property t-SNE skipped because no generated method has property vectors.")
        else:
            props_by_group_aligned = align_property_dims(props_by_group, truncate=True)
            prop_emb, prop_labels = make_tsne(
                props_by_group_aligned, seed=args.seed, perplexity=args.perplexity, max_iter=args.tsne_iter
            )
            save_tsne_plot(
                prop_emb,
                prop_labels,
                os.path.join(args.out_dir, "tsne_property"),
                title="Property distribution: train vs. generated",
            )

    print("Done.")


if __name__ == "__main__":
    main()
