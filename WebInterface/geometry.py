"""Safe, NumPy-only access to saved lattices; no model imports or pickle loading."""
import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np


def read_lattice(source):
    """Preserve source edges and report checks without silently repairing a graph."""
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with zipfile.ZipFile(source) as archive:
        if sum(item.file_size for item in archive.infolist()) > 20 * 1024 * 1024:
            raise ValueError("The uncompressed lattice exceeds 20 MB.")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("The archive contains duplicate fields.")
        for name in ["frac_coords", "edge_index", "lengths", "angles", "y_pred"]:
            member = name + ".npy"
            if member not in names:
                continue
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                if dtype.kind not in "iuf" or math.prod(shape) * dtype.itemsize > archive.getinfo(member).file_size - stream.tell():
                    raise ValueError("Geometry arrays must contain numeric data with valid dimensions.")
    if hasattr(source, "seek"):
        source.seek(0)
    with np.load(source, allow_pickle=False) as data:
        required = {"frac_coords", "edge_index", "lengths", "angles"}
        if not required.issubset(data.files):
            raise ValueError("Expected frac_coords, edge_index, lengths, and angles in the NPZ file.")
        coords = np.asarray(data["frac_coords"], dtype=float)
        edges = data["edge_index"]
        lengths = np.asarray(data["lengths"], dtype=float).reshape(-1)
        angles = np.asarray(data["angles"], dtype=float).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3 or not 1 <= len(coords) <= 2000:
            raise ValueError("Coordinates must contain 1–2000 rows of three numbers.")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] > 10000:
            raise ValueError("edge_index must have shape (2, E), with at most 10000 edges.")
        if not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("Edge indices must be integers.")
        if edges.size and (edges.min() < 0 or edges.max() >= len(coords)):
            raise ValueError("An edge references a node outside the coordinate array.")
        if lengths.shape != (3,) or angles.shape != (3,):
            raise ValueError("Lengths and angles must each contain three numbers.")
        if not all(np.isfinite(a).all() for a in [coords, lengths, angles]):
            raise ValueError("Geometry contains a non-finite number.")
        if (lengths <= 0).any() or (angles <= 0).any() or (angles >= 180).any():
            raise ValueError("Cell lengths must be positive and angles between 0 and 180 degrees.")
        if (np.abs(coords) > 1e6).any() or (lengths > 1e6).any():
            raise ValueError("Geometry exceeds the supported numeric range.")
        # Same orientation as utils.mat_utils.lattice_params_to_matrix, without
        # importing Torch and torch_scatter into the browsing-only process.
        alpha, beta, gamma = np.radians(angles)
        star = (np.cos(alpha) * np.cos(beta) - np.cos(gamma)) / (np.sin(alpha) * np.sin(beta))
        if abs(star) >= 1:
            raise ValueError("The cell angles define a degenerate or impossible cell.")
        a, b, c = lengths
        basis = np.array([[a * np.sin(beta), 0, a * np.cos(beta)],
                          [-b * np.sin(alpha) * star, b * np.sin(alpha) * np.sqrt(1 - star**2), b * np.cos(alpha)],
                          [0, 0, c]])
        cart = coords @ basis
        if not np.isfinite(cart).all():
            raise ValueError("Cell geometry could not be represented as finite coordinates.")
        unique = np.unique(np.sort(edges.T, axis=1), axis=0)
        adjacent = [set() for _ in coords]
        for left, right in unique:
            if left != right:
                adjacent[left].add(int(right))
                adjacent[right].add(int(left))
        unseen = set(range(len(coords)))
        components = 0
        while unseen:
            components += 1
            stack = [unseen.pop()]
            while stack:
                new = adjacent[stack.pop()] & unseen
                unseen.difference_update(new)
                stack.extend(new)
        strut_lengths = np.linalg.norm(cart[unique[:, 0]] - cart[unique[:, 1]], axis=1)
        warnings = []
        if ((coords < -1e-6) | (coords > 1 + 1e-6)).any():
            warnings.append("Some fractional coordinates lie outside [0, 1]; they are displayed as stored.")
        if len(unique) < edges.shape[1]:
            warnings.append("Repeated or reversed edges are drawn once; original edges are preserved in exports.")
        predictions = None
        if "y_pred" in data.files:
            values = np.asarray(data["y_pred"], dtype=float).reshape(-1)
            if len(values) == 12 and np.isfinite(values).all():
                predictions = values.tolist()
        # prop_list is conditioning, never a prediction; legacy objects stay unread.
        fingerprint = hashlib.sha256()
        for array in [coords, edges, lengths, angles]:
            fingerprint.update(array.tobytes())
        return dict(frac_coords=coords.tolist(), cart_coords=cart.tolist(),
                    edge_index=edges.tolist(), edges=unique.tolist(),
                    lengths=lengths.tolist(), angles=angles.tolist(), basis=basis.tolist(),
                    nodes=len(coords), struts=len(unique), stored_edges=edges.shape[1],
                    degrees=[len(neighbors) for neighbors in adjacent], components=components,
                    zero_length_struts=int(np.sum(strut_lengths < 1e-9)),
                    strut_lengths=strut_lengths.tolist(), predictions=predictions,
                    warnings=warnings, fingerprint=fingerprint.hexdigest())
