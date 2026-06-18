"""
bipartite_pore_graph.py
=======================
Builds a bipartite atom-pore DGL heterograph from a jarvis Atoms object.

Graph structure:
    N = N1 u N2
    N1 = atom nodes  (same atoms as ALIGNN's crystal graph)
    N2 = pore nodes  (one per physical cavity, from Voronoi + radius-aware clustering)
    Edges ONLY between N1 and N2 -- never atom-atom, never pore-pore.

An edge (pore p, atom a) exists when:
    dist(pore_center_p, atom_position_a) < max(R_pore_p + R_vdW_a, MIN_SEARCH)

Node features:
    Atom nodes : 92-dim cgcnn vector from jarvis (same as ALIGNN's atom graph)
    Pore nodes : [pore_radius (Å), cluster_size (n raw Voronoi nodes merged)]

Edge features:
    d : distance from pore center to atom (Å), same for both directions

Main entry point:
    G = build_bipartite_pore_graph(atoms)

Returns None if the structure has no accessible pores (dense MOFs) or if
voro++ fails for unusual unit cells).
"""

import os
import tempfile
from collections import defaultdict

import numpy as np
import torch
import dgl
from scipy.spatial.distance import cdist
from jarvis.core.specie import get_node_attributes
from pyzeo.netstorage import AtomNetwork


# Constants

PROBE_RADIUS    = 1.2   # Å — CO2 probe radius used in write_to_XYZ
MIN_PORE_RADIUS = 1.2   # Å — discard artefact pores smaller than this
MIN_SEARCH      = 5.0   # Å — minimum pore-to-atom search radius (fixes small
                        #     pores near cell boundaries that get 0 wall atoms)

# Alvarez 2013 van der Waals radii for elements present in hMOF dataset
RADII = {
    'H':  1.20, 'C':  1.77, 'N':  1.66, 'O':  1.50,
    'F':  1.46, 'Al': 2.25, 'Si': 2.19, 'P':  1.90,
    'S':  1.89, 'Cl': 1.82, 'Br': 1.86, 'Zn': 2.39,
    'Cu': 2.38, 'Co': 2.40, 'Ni': 2.40, 'Fe': 2.44,
    'Mn': 2.45, 'Cr': 2.45, 'V':  2.42, 'Ti': 2.46,
    'Cd': 2.49, 'Zr': 2.52, 'Mo': 2.45, 'In': 2.43,
}


# write pyzeo input files

def _atoms_to_cssr(atoms, path):
    """Write a jarvis Atoms object to CSSR format for pyzeo.
    Note: pyzeo 0.1.7 requires the path to be passed as bytes (.encode()).
    """
    lat = atoms.lattice_mat
    elements = atoms.elements
    coords = atoms.frac_coords
    a, b, c = lat[0][0], lat[1][1], lat[2][2]
    with open(path, 'w') as f:
        f.write(f'                {a:.4f}  {b:.4f}  {c:.4f}\n')
        f.write(f'   90.0000   90.0000   90.0000   SPGR =  1 P 1\n')
        f.write(f' {len(elements)} 0\n')
        f.write(f' 0 {atoms.composition.formula}\n')
        for i, (el, fc) in enumerate(zip(elements, coords)):
            f.write(
                f'{i+1} {el} {fc[0]:.6f} {fc[1]:.6f} {fc[2]:.6f}'
                f'  0  0  0  0  0  0  0  0 0.000\n'
            )


def _write_rad_file(atoms, path):
    """Write a pyzeo radii file: one line per unique element with its vdW radius."""
    with open(path, 'w') as f:
        for el in set(atoms.elements):
            f.write(f'{el} {RADII.get(el, 1.5)}\n')


# run pyzeo Voronoi decomposition

def _run_pyzeo(atoms):
    """Run pyzeo on a jarvis Atoms object.

    Returns:
        positions : np.ndarray (N, 3) coords of accessible Voronoi nodes
        radii     : np.ndarray (N,)   radius at each node (largest inscribed sphere)

    Returns (None, None) if no accessible pores exist.
    Raises on voro++ errors, should catch Exception.
    """
    with tempfile.NamedTemporaryFile(suffix='.cssr', delete=False, mode='w') as tmp:
        cssr_path = tmp.name
    rad_path = cssr_path.replace('.cssr', '.rad')
    xyz_path = cssr_path.replace('.cssr', '.xyz')

    try:
        _atoms_to_cssr(atoms, cssr_path)
        _write_rad_file(atoms, rad_path)

        atm_ntw = AtomNetwork.read_from_CSSR(
            cssr_path.encode(),
            rad_flag=True,
            rad_file=rad_path.encode(),
        )
        vornet, _edge_centers, _face_centers = atm_ntw.perform_voronoi_decomposition()
        vornet.write_to_XYZ(xyz_path, PROBE_RADIUS)

        with open(xyz_path) as f:
            lines = f.readlines()

        n_nodes = int(lines[0].strip())
        if n_nodes == 0:
            return None, None

        nodes = []
        for line in lines[2:2 + n_nodes]:
            parts = line.split()
            x, y, z, r = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            nodes.append((x, y, z, r))

        positions = np.array([[n[0], n[1], n[2]] for n in nodes])
        radii     = np.array([n[3] for n in nodes])
        return positions, radii

    finally:
        for p in [cssr_path, rad_path, xyz_path]:
            if os.path.exists(p):
                os.unlink(p)


# radius-aware clustering

def _radius_aware_clustering(positions, radii):
    """Collapse raw Voronoi nodes into physically distinct pores.

    Two nodes belong to the same pore if one sits inside the other's sphere:
        dist(i, j) < max(radius_i, radius_j)

    Keeps the node with the largest radius per cluster.
    Uses Union-Find with path compression.
    """
    n = len(positions)
    if n == 0:
        return positions, radii, []

    D         = cdist(positions, positions)
    R_max     = np.maximum.outer(radii, radii)
    same_pore = (D < R_max) & (D > 0)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if same_pore[i, j]:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    cluster_pos, cluster_rad, cluster_sizes = [], [], []
    for members in clusters.values():
        members = np.array(members)
        best    = members[radii[members].argmax()]
        cluster_pos.append(positions[best])
        cluster_rad.append(radii[best])
        cluster_sizes.append(len(members))

    return np.array(cluster_pos), np.array(cluster_rad), cluster_sizes


# build bipartite edges

def _build_bipartite_edges(cluster_pos, cluster_rad, atom_positions, atom_elements):
    """Find pore–atom edges.

    Edge criterion:
        dist(pore_center_p, atom_position_a) < max(R_pore_p + R_vdW_a, MIN_SEARCH)

    The MIN_SEARCH floor ensures small pores near cell boundaries
    still find their wall atoms.
    """
    atom_radii = np.array([RADII.get(el, 1.5) for el in atom_elements])
    D_pa       = cdist(cluster_pos, atom_positions)
    threshold = np.maximum(
        cluster_rad[:, None] + atom_radii[None, :] + 0.1,   # +0.1 buffer
        MIN_SEARCH,
    )
    pore_src, atom_dst = np.where(D_pa < threshold)
    edge_distances     = D_pa[pore_src, atom_dst]
    return pore_src, atom_dst, edge_distances


# atom node features

def _atom_features(atom_elements):
    """92-dim cgcnn feature vector per atom -- same encoding ALIGNN uses."""
    return np.array(
        [list(get_node_attributes(el, atom_features='cgcnn')) for el in atom_elements],
        dtype=np.float32,
    )  # shape (N_atoms, 92)


# Main entry point

def build_bipartite_pore_graph(atoms):
    """Build a bipartite atom-pore DGL graph from a jarvis Atoms object.

    Returns None if:
      - The structure has no accessible pores (dense MOF)
      - voro++ fails (e.g. "Periodic cell computation failed")
      - All pores are artefacts smaller than MIN_PORE_RADIUS

    Graph schema:
        G.nodes['atom'].data['h']     : (N_atoms, 92)  cgcnn element features
        G.nodes['pore'].data['h']     : (N_pores, 2)   [radius (Å), cluster_size]
        G.edges['wall'].data['d']     : (E, 1)          pore-center to atom dist (Å)
        G.edges['wall_rev'].data['d'] : (E, 1)          same, reverse direction
    """
    lat = atoms.lattice_mat
    if lat[0][0] < 0.1 or lat[1][1] < 0.1 or lat[2][2] < 0.1:
        return None  # non-orthogonal cell -- skip safely
    
    # pyzeo: catch ALL failures including "Periodic cell computation failed"
    try:
        positions, radii = _run_pyzeo(atoms)
    except Exception:
        return None  # voro++ failed, treat as no pore information

    if positions is None:
        return None  # no accessible pores

    # radius-aware clustering
    cluster_pos, cluster_rad, cluster_sizes = _radius_aware_clustering(positions, radii)

    # filter artefact pores
    keep          = cluster_rad >= MIN_PORE_RADIUS
    cluster_pos   = cluster_pos[keep]
    cluster_rad   = cluster_rad[keep]
    cluster_sizes = [s for s, k in zip(cluster_sizes, keep) if k]

    if len(cluster_pos) == 0:
        return None

    # bipartite edges
    atom_positions = np.array(atoms.cart_coords)
    atom_elements  = atoms.elements

    pore_src, atom_dst, edge_distances = _build_bipartite_edges(
        cluster_pos, cluster_rad, atom_positions, atom_elements
    )

    N_atoms = len(atom_elements)
    N_pores = len(cluster_pos)

    # node features
    atom_feats = _atom_features(atom_elements)
    pore_feats = np.stack([
        cluster_rad,
        np.array(cluster_sizes, dtype=np.float32),
    ], axis=1)

    # assemble DGL graph
    G = dgl.heterograph(
        {
            ('pore', 'wall',     'atom'): (pore_src.tolist(), atom_dst.tolist()),
            ('atom', 'wall_rev', 'pore'): (atom_dst.tolist(), pore_src.tolist()),
        },
        num_nodes_dict={'atom': N_atoms, 'pore': N_pores},
    )

    G.nodes['atom'].data['h'] = torch.tensor(atom_feats, dtype=torch.float32)
    G.nodes['pore'].data['h'] = torch.tensor(pore_feats, dtype=torch.float32)

    dist_tensor = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1)
    G.edges['wall'].data['d']     = dist_tensor
    G.edges['wall_rev'].data['d'] = dist_tensor

    return G


# Quick test

if __name__ == '__main__':
    from jarvis.db.figshare import data as jdata
    from jarvis.core.atoms import Atoms

    print('Loading hMOF dataset...')
    raw = jdata('hmof')

    for idx in [0, 100, 500, 1000]:
        entry = raw[idx]
        atoms = Atoms.from_dict(entry['atoms'])
        G     = build_bipartite_pore_graph(atoms)

        if G is None:
            print(f'[{idx}] {entry["id"]:20s}  -- no pore graph')
            continue

        N_atoms = G.num_nodes('atom')
        N_pores = G.num_nodes('pore')
        N_edges = G.num_edges('wall')
        radii   = G.nodes['pore'].data['h'][:, 0].tolist()

        print(
            f'[{idx}] {entry["id"]:20s}'
            f'  atoms={N_atoms:4d}'
            f'  pores={N_pores:2d}'
            f'  edges={N_edges:4d}'
            f'  pore_radii={[round(r,1) for r in radii]}'
        )
