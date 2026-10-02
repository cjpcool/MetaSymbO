import re
import torch
from utils.lattice_utils import classify_nodes_with_geometry
import numpy as np

import re
import torch
import warnings

def parse_graph(text: str):
    """
    Parse a node/edge description and return:
      z           – Tensor[N]        (long)
      coords      – Tensor[N, 3]     (float32)
      edge_index  – Tensor[2, M]     (long)
      batch       – Tensor[N]        (long, all zeros)
      lengths     – Tensor[1, 3]     (float32)
      angles      – Tensor[1, 3]     (float32)
      num_atoms   – Tensor[1]        (long)
    Supports sections:
      - Node number: <int>              (optional)
      - Node coordinates:
      - Edges:
      - Lattice Lengths:
      - Lattice Angles:
    """

    # Normalize to make section header searches robust.
    low = text.lower()

    def _sec_span(key: str):
        """Return (start_idx, end_idx_of_header_label) for a header key in lower case text."""
        k = key.lower()
        i = low.find(k)
        if i == -1:
            return None
        return i, i + len(k)

    # Compute explicit section ranges
    # Order we’ll try to parse blocks: coords, edges, lengths, angles.
    h_coords = _sec_span("Node coordinates:")
    h_edges  = _sec_span("Edges:")
    h_len    = _sec_span("Lattice Lengths:")
    h_ang    = _sec_span("Lattice Angles:")

    # Helper to slice a block from the original text based on two headers.
    def _block(after_hdr_span, next_hdr_span):
        if after_hdr_span is None:
            return ""
        start = after_hdr_span[1]
        end = len(text) if next_hdr_span is None else next_hdr_span[0]
        return text[start:end].strip()

    # Determine the “next header” for each section based on their order of appearance.
    # Build a list of present headers with their starting positions to order them.
    hdrs = []
    if h_coords: hdrs.append(("coords", h_coords[0], h_coords))
    if h_edges:  hdrs.append(("edges",  h_edges[0],  h_edges))
    if h_len:    hdrs.append(("lengths",h_len[0],    h_len))
    if h_ang:    hdrs.append(("angles", h_ang[0],    h_ang))
    hdrs.sort(key=lambda x: x[1])  # by appearance

    # Map each header to the next header’s span (or None if last)
    next_map = {}
    for idx, (name, _, span) in enumerate(hdrs):
        next_map[name] = hdrs[idx+1][2] if idx+1 < len(hdrs) else None

    coord_block = _block(h_coords, next_map.get("coords"))
    edge_block  = _block(h_edges,  next_map.get("edges"))
    len_block   = _block(h_len,    next_map.get("lengths"))
    ang_block   = _block(h_ang,    next_map.get("angles"))

    # --- helper: floats/ints in a line --------------------------------------
    num_pat = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')

    # Parse coordinates -------------------------------------------------------
    coord_lines = [ln for ln in coord_block.splitlines() if num_pat.search(ln)]
    coords = [
        [float(x) for x in num_pat.findall(line)[:3]]
        for line in coord_lines
    ]
    if not coords:
        raise ValueError("No coordinates parsed. Ensure 'Node coordinates:' section exists.")
    coords = torch.tensor(coords, dtype=torch.float32)  # [N, 3]

    # Parse edges -------------------------------------------------------------
    edge_lines = [ln for ln in edge_block.splitlines() if num_pat.search(ln)]
    edges = [
        [int(float(x)) for x in num_pat.findall(line)[:2]]
        for line in edge_lines
    ]
    if not edges:
        warnings.warn("No edges parsed. 'Edges:' section empty or missing?", RuntimeWarning)
        edges = []
    edges = torch.tensor(edges, dtype=torch.long) if len(edges) > 0 else torch.empty((0,2), dtype=torch.long)
    edge_max = edges.max().item()
    edge_min = edges.min().item()
    if edge_max == coords.shape[0] and edge_min == 1:
        edges -= 1  # convert 1-based to 0-based indexing
    elif edge_max >= coords.shape[0] or edge_min < 0:
        raise ValueError(f"Edge indices out of range: min {edge_min}, max {edge_max}, but have {coords.shape[0]} nodes.")
    edge_index = edges.T  # [2, M]

    # Parse lattice lengths and angles ---------------------------------------
    def _parse_triplet(block_text, default_vals):
        nums = num_pat.findall(block_text)
        if len(nums) >= 3:
            trip = [float(nums[0]), float(nums[1]), float(nums[2])]
        else:
            trip = list(default_vals)
        return torch.tensor([trip], dtype=torch.float32)  # [1, 3]

    lengths = _parse_triplet(len_block, default_vals=(1.0, 1.0, 1.0))
    angles  = _parse_triplet(ang_block, default_vals=(90.0, 90.0, 90.0))

    # Construct other elements -----------------------------------------------
    N = coords.shape[0]
    num_atoms = torch.tensor([N], dtype=torch.long)
    batch = torch.zeros((N,), dtype=torch.long)

    # Optional: derive node labels if helper is available
    try:
        node_labels = classify_nodes_with_geometry(coords, edge_index)  # user-provided elsewhere
        z = torch.argmax(node_labels, dim=-1).to(torch.long) + 1
    except Exception:
        warnings.warn("No geometry classifier available; using default label 1 for all nodes.")
        z = torch.ones((N,), dtype=torch.long)

    return z, coords, edge_index, batch, lengths, angles, num_atoms





def graph_to_text(coords: torch.Tensor, edges: torch.Tensor) -> str:
    """
    Transforms node coordinates and edge list tensors into formatted text.

    Args:
        coords (torch.Tensor): Tensor of shape (n, 3) for node coordinates.
        edges (torch.Tensor): Tensor of shape (2, m) for edges (start and end indices).

    Returns:
        str: Formatted text representation.
    """
    n = coords.size(0)
    lines = []
    lines.append(f"Node number: {n}")
    lines.append("Node coordinates:")
    for x, y, z in coords.detach().cpu().tolist():
        lines.append(f"({x}, {y}, {z})")
    lines.append("")  # Blank line before edges
    lines.append("Edges:")
    # Assume edges is of shape (2, m)
    start_nodes, end_nodes = edges.tolist()
    for u, v in zip(start_nodes, end_nodes):
        lines.append(f"({u}, {v})")
    return "\n".join(lines)


def construct_supervisor_input(prompt, structure, lattice_lengths=None, lattice_angles=None, properties=None):
    """
    Constructs the input for the supervisor model by combining the prompt,
    structure, and optional properties.

    Args:
        prompt (str): The prompt text.
        structure (str): The structure text.
        properties (str, optional): The properties text. Defaults to None.

    Returns:
        str: The combined input string.
    """
    input_text = f"Prompt: {prompt}\n\nStructure: {structure}\n"
    if lattice_lengths is None and lattice_angles is None:
        input_text += "\nLattice lengths: [1.0, 1.0, 1.0]\nLattice angles: [90.0, 90.0, 90.0]\n"
    else:
        input_text += f"\nLattice lengths: {lattice_lengths}\nLattice angles: {lattice_angles}"
    if properties is not None:
        input_text += "\nProperties:"
        for key, value in properties.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
            input_text += f"\n{key}: {value}"
    return input_text




def find_closest_structure(
    dataset,
    improved_properties: torch.Tensor,
    batch_size: int = 10000,   # tune for your GPU / RAM budget
    p: int = 2,                 # 2 = Euclidean, 1 = Manhattan, etc.
    property_vectors: torch.Tensor = None,
):
    if property_vectors is not None:
        query = torch.as_tensor(improved_properties, dtype=torch.float32, device=property_vectors.device).view(1, -1)
        best_distance, best_index = float('inf'), -1
        for start in range(0, len(property_vectors), batch_size):
            distances = torch.linalg.vector_norm(property_vectors[start:start + batch_size] - query, ord=p, dim=1)
            distance, index = distances.min(dim=0)
            if distance.item() < best_distance:
                best_distance, best_index = distance.item(), start + index.item()
        return [best_index], best_distance

    query = torch.as_tensor(improved_properties, dtype=torch.float32).view(1, -1)

    # Make sure we run on the same device as the data (or stick to CPU)
    query = query.to(next(iter(dataset)).y.device)

    best_dist = np.inf
    best_idx  = -1

    selected_idx = []

    # Stream through the dataset in chunks
    batch = []
    start_idx = 0
    for i, data in enumerate(dataset):
        batch.append(data.y.view(-1))           # flatten to (12,)
        if len(batch) == batch_size or i == len(dataset) - 1:
            Ys = torch.stack(batch)             # (B, 12) on same device
            diff = Ys - query                   # broadcast (B, 12)
            dists = torch.linalg.vector_norm(diff, ord=p, dim=1)   # (B,)
            min_val, min_pos = torch.min(dists, dim=0)

            if min_val.item() < best_dist:
                best_dist = min_val.item()
                best_idx  = start_idx + min_pos.item()

            # reset for next chunk
            batch.clear()
            start_idx = i + 1

    return [best_idx], best_dist


