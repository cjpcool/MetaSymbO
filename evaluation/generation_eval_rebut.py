import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')


from scipy.spatial.distance import cdist
from torch_cluster import radius_graph
from tqdm import tqdm

from datasets.dataset_truss import LatticeStiffness, LatticeModulus
from scipy.optimize import linear_sum_assignment
from eval_utils import cart_to_frac_coords
from typing import List

import torch
import numpy as np

import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from datasets.baseDataset import LatticeTruss
from eval_utils import lattice_params_to_matrix, frac_to_cart_coords
from itertools import combinations
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.utils import shuffle

from visualization.vis import visualizeLattice


MAX_NODE_NUM = 100

def pad_to_max_length(fps_list,max_length, padding_value=float('inf')):
    # Pad each fingerprint to the max_length with the specified padding_value
    padded_fps = [np.pad(fp, ((0,max_length - len(fp)), (0,0)), 'constant', constant_values=(padding_value)) for fp in fps_list]
    return np.array(padded_fps)


class LatticeEvaluatorMaster():
    def __init__(self,
                 cart_coords: List[np.ndarray] = None,
                 frac_coords: List[np.ndarray] = None,
                 node_types: List[np.ndarray] = None,
                 edges: List[np.ndarray] = None,
                 lattice_vectors: List[np.ndarray] = None,
                 ):
        self.cart_coords = [] if cart_coords is None else cart_coords
        self.frac_coords = [] if frac_coords is None else frac_coords
        self.node_types = [] if node_types is None else node_types
        self.edges = [] if edges is None else edges
        self.lattice_vectors = [] if lattice_vectors is None else lattice_vectors


    def eval_graph_validity(self, **kwargs):
        return NotImplementedError

    def eval_condition_effectiveness(self, **kwargs):
        return NotImplementedError

    def obtain_stiffness(self, **kwargs):
        return NotImplementedError

    def eval_diversity(self):
        return NotImplementedError




class LatticeEvaluator(LatticeEvaluatorMaster):
    def __init__(self,
                 test_datset: LatticeTruss=None,
                 eval_file_path: str=None,
                 cart_coords: List[np.ndarray] = None,
                 frac_coords: List[np.ndarray] = None,
                 node_types: List[np.ndarray] = None,
                 edges: List[np.ndarray] = None,
                 lattice_vectors: List[np.ndarray] = None,
                 cluster_size: int = 50,
                 data_size_for_eval: int = 2000,
                 central_symmetry_error_bar=0.1,
                 periodic_error_bar=0.1,
                 diversity_error_bar=0.1):

        super().__init__(
            cart_coords,
            frac_coords,
            node_types,
            edges,
            lattice_vectors
        )
        self.cond_prop = []
        if eval_file_path is not None:
            self.__read_eval_data(eval_file_path)
        else:
            self.cart_coords = cart_coords
            self.frac_coords = frac_coords
            self.node_types = node_types
            self.edges = edges
            self.lattice_vectors = lattice_vectors




        self.central_symmetry_error_bar = central_symmetry_error_bar
        self.periodic_error_bar = periodic_error_bar
        self.diversity_error_bar = diversity_error_bar



        self.test_dataset = test_datset
        self.cluster_size = cluster_size
        if data_size_for_eval is not None:
            self.data_size_for_eval = data_size_for_eval
        else:
            self.data_size_for_eval = len(test_datset)
        assert self.cluster_size <= self.data_size_for_eval, 'data_size_for_eval < cluster_size'


    @staticmethod
    def _npz_get_optional(lattice_npz, key, default=None):
        """Safely read optional fields from an npz file.

        CDVAE generations may save missing fields as 0-D object arrays, e.g.
        edge_index=array(None, dtype=object) or prop_list=array(None, dtype=object).
        This helper converts those cases to Python None.
        """
        if key not in lattice_npz.files:
            return default
        value = lattice_npz[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            value = value.item()
        if value is None:
            return default
        return value

    @staticmethod
    def _normalize_edge_index(edge_index, num_nodes):
        """Return a valid edge index with shape (2, E), or None if unavailable.

        This keeps MetaSymbO files unchanged while preventing CDVAE files with
        edge_index=None from crashing the loader. Distance-to-train evaluation
        does not require edges; validity/connectivity metrics will safely treat
        missing edges as invalid through existing checks.
        """
        if edge_index is None:
            return None

        edge_index = np.asarray(edge_index)
        if edge_index.dtype == object and edge_index.shape == ():
            edge_index = edge_index.item()
            if edge_index is None:
                return None
            edge_index = np.asarray(edge_index)

        if edge_index.size == 0:
            return None

        if edge_index.ndim != 2:
            return None

        # Accept both (2, E) and (E, 2).
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.T
        if edge_index.shape[0] != 2:
            return None

        edge_index = edge_index.astype(np.int64, copy=False)
        valid_mask = (edge_index[0] >= 0) & (edge_index[0] < num_nodes) & \
                     (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        edge_index = edge_index[:, valid_mask]
        if edge_index.shape[1] == 0:
            return None
        return edge_index

    def __read_eval_data(self, eval_file_path):
        if os.path.isfile(eval_file_path):
            file_paths = [eval_file_path]
        else:
            file_paths = [
                os.path.join(eval_file_path, file_name)
                for file_name in sorted(os.listdir(eval_file_path))
                if file_name.endswith('.npz')
            ]

        for full_path in file_paths:
            lattice_npz = np.load(full_path, allow_pickle=True)

            # CDVAE files use lengths/angles + frac_coords, but may not contain
            # a precomputed lattice vector. MetaSymbO files with `vector` are
            # still supported.
            lattice_lengths = lattice_npz['lengths']
            lattice_angles = lattice_npz['angles']
            lattice_vector = self._npz_get_optional(lattice_npz, 'vector')
            if lattice_vector is None:
                lattice_vector = lattice_params_to_matrix(
                    lattice_lengths[0], lattice_lengths[1], lattice_lengths[2],
                    lattice_angles[0], lattice_angles[1], lattice_angles[2])
            lattice_vector = np.asarray(lattice_vector)
            self.lattice_vectors.append(lattice_vector)

            frac_coord = self._npz_get_optional(lattice_npz, 'frac_coords')
            cart_coord = self._npz_get_optional(lattice_npz, 'cart_coords')

            if frac_coord is None and cart_coord is None:
                raise KeyError(f'{full_path} must contain either frac_coords or cart_coords.')

            if frac_coord is None:
                cart_coord = np.asarray(cart_coord)
                frac_coord = cart_to_frac_coords(cart_coord, lattice_vector, len(cart_coord))
            else:
                frac_coord = np.asarray(frac_coord)

            num_atoms = len(frac_coord)
            if cart_coord is None:
                cart_coord = frac_to_cart_coords(frac_coord, lattice_vector, num_atoms)
            else:
                cart_coord = np.asarray(cart_coord)

            self.frac_coords.append(frac_coord)
            self.cart_coords.append(cart_coord)

            atom_types = self._npz_get_optional(lattice_npz, 'atom_types')
            if atom_types is None:
                atom_types = self._npz_get_optional(lattice_npz, 'node_types')
            self.node_types.append(None if atom_types is None else np.asarray(atom_types))

            edge_index = self._npz_get_optional(lattice_npz, 'edge_index')
            edge_index = self._normalize_edge_index(edge_index, num_atoms)
            self.edges.append(edge_index)

            cond_prop = self._npz_get_optional(lattice_npz, 'prop_list')
            if cond_prop is not None:
                self.cond_prop.append(np.asarray(cond_prop))


    def evaluate_all_uncondition_generation(self):
        periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio = self.eval_graph_validity()
        cov_r, cov_p = self.eval_diversity()
        repeated_ratio = self.compute_repeated_ratio(self.cart_coords, 0.1)
        print(f"repeated_ratio: {repeated_ratio}")


    def eval_condition_effectiveness(self):
        eval_number = len(self.cart_coords)
        distances = []
        assert len(self.cart_coords) == len(self.cond_prop)


        idx_node_num  = list(range(len(self.test_dataset)))
        right = self.data_size_for_eval
        selected_idx = shuffle(idx_node_num)[:right]
        selected_data_for_eval = self.test_dataset.copy(selected_idx)

        train_x = [data.cart_coords for data in selected_data_for_eval]
        train_y = self.test_dataset.data.y[selected_idx]

        neigh_y = NearestNeighbors(n_neighbors=self.cluster_size, metric='euclidean')
        neigh_y.fit(train_y)


        for i in tqdm(range(eval_number), desc='eval_condition_effectiveness'):
            x_gen = self.cart_coords[i]
            y_cond = self.cond_prop[i]

            dist = self.condition_effectiveness(y_cond, x_gen, train_x, neigh_y)
            # print(dist)

            distances.append(dist)
        print(f'Conditional Effectiveness:{np.mean(distances)}')


    @staticmethod
    def condition_effectiveness(y_cond, x_gen, train_x, neigh_y):

        _, cluster_gen_idx = neigh_y.kneighbors(y_cond.reshape(1,-1))

        # neigh_x = NearestNeighbors(n_neighbors=cluster_size, metric='euclidean')
        # neigh_x.fit(train_x)
        # _, cluster_gen_idx = neigh_x.kneighbors(x_gen)
        cluster_gen_x = [train_x[idx.item()] for idx in cluster_gen_idx[0]]
        # cluster_gen_y = train_y[cluster_gen_idx]
        dist_list = []
        for clustered_x in cluster_gen_x:
            dist = LatticeEvaluator.compute_uc_dist_Hungary_alg(x_gen, clustered_x)
            dist_list.append(dist.mean())


        return np.min(dist_list)

    @staticmethod
    def compute_repeated_ratio(coords, dist_threshold=.05):
        """
        Compute the repeated ratio of a graph.

        Args:
            coords (torch.Tensor): Node coordinates.
            edge_index (torch.Tensor): Edge indices.
            batch (torch.Tensor): Batch indices.

        Returns:
            float: Repeated ratio.
        """
        # Implement the logic to compute the repeated ratio
        # This is a placeholder implementation
        repeated_num = 0
        for i in tqdm(range(len(coords))):
            for j in range(i + 1, len(coords)):
                if coords[i].shape != coords[j].shape:
                    continue
                # Calculate the distance between the two nodes
                dist = cdist(coords[i], coords[j], 'euclidean')
                max_pair_dist = dist.min(axis=1).max()
                # Check if the distance is less than a threshold (e.g., 1.0)
                if max_pair_dist < dist_threshold:
                    repeated_num += 1
                    break

        repeated_ratio = repeated_num / len(coords)
        print('repeated_num', repeated_num)
        print('len(coords)', len(coords))
        return repeated_ratio


    def eval_graph_validity(self, use_fractional_coords=False):
        if use_fractional_coords:
            coords = self.frac_coords
            lattice_lengths = torch.ones(3).float()
            lattice_angles = torch.ones(3).float() * 90
            lattice_vectors = [lattice_params_to_matrix(lattice_lengths[0],lattice_lengths[1],lattice_lengths[2],
                                                  lattice_angles[0], lattice_angles[1], lattice_angles[2]) for _ in self.lattice_vectors]
        else:
            coords = self.cart_coords
            lattice_vectors = self.lattice_vectors
        return self.graph_validity(coords, self.edges, lattice_vectors)

    def eval_diversity(self):
        train_x =  [x.cart_coords for x in self.test_dataset]

        metrics_dict, _ = self.compute_cov(self.cart_coords, train_x, self.diversity_error_bar)
        print(metrics_dict)
        return metrics_dict['cov_recall'], metrics_dict['cov_precision']

    @staticmethod
    def _coord_to_numpy(coord):
        """Convert one coordinate array/tensor to a NumPy array."""
        if isinstance(coord, torch.Tensor):
            coord = coord.detach().cpu().numpy()
        return np.asarray(coord)

    @staticmethod
    def _dataset_to_cart_coords(dataset):
        """Extract Cartesian coordinates from a lattice dataset/list."""
        return [LatticeEvaluator._coord_to_numpy(data.cart_coords) for data in dataset]

    def eval_distance_to_train(self,
                               train_dataset: LatticeTruss = None,
                               train_coords: List[np.ndarray] = None,
                               num_train_structure: int = None,
                               seed: int = 42,
                               return_per_sample: bool = False):
        """
        Evaluate generated structures by nearest-neighbor distance to the training set.

        For each generated structure g, we compute its structure distance to every
        training structure t using the same unit-cell distance used in coverage:
            d(g, t) = mean_i min_j ||g_i - t_j||_2.
        We then keep the nearest training distance for each generated sample.

        Returns:
            avg_dist_to_train: mean nearest-training distance over generated samples.
            min_dist_to_train: smallest nearest-training distance among generated samples.

        Larger values indicate generated structures are farther from the training
        set under this coordinate-level metric.
        """
        if train_coords is None:
            if train_dataset is None:
                train_dataset = self.test_dataset
            assert train_dataset is not None, \
                'Please provide either train_dataset or train_coords.'
            train_coords = self._dataset_to_cart_coords(train_dataset)
        else:
            train_coords = [self._coord_to_numpy(coord) for coord in train_coords]

        if num_train_structure is not None and num_train_structure < len(train_coords):
            rng = np.random.default_rng(seed)
            selected_idx = rng.choice(len(train_coords), size=num_train_structure, replace=False)
            train_coords = [train_coords[idx] for idx in selected_idx]

        assert len(self.cart_coords) > 0, 'No generated structures are loaded.'
        assert len(train_coords) > 0, 'No training structures are provided.'

        gen_to_train_min_dist = []
        for gen_idx, gen_coord in enumerate(tqdm(self.cart_coords, desc='Computing Gen-Train Dist')):
            gen_coord = self._coord_to_numpy(gen_coord)

            if False:
                lattice_vector = None
                if self.lattice_vectors is not None and gen_idx < len(self.lattice_vectors):
                    lattice_vector = self._coord_to_numpy(self.lattice_vectors[gen_idx])

                periodic_valid = False
                if lattice_vector is not None:
                    periodic_valid = LatticeEvaluator.is_periodic_necessary_condition(
                        gen_coord,
                        lattice_vector.reshape(3, 3),
                        error_bar=self.periodic_error_bar,
                    )

                symmetry_score = LatticeEvaluator.central_symmetry(
                    gen_coord,
                    error_bar=self.central_symmetry_error_bar,
                )
                symmetry_valid = symmetry_score > 0.

                if not (periodic_valid and symmetry_valid):
                    continue
            dist_to_train = []
            for train_coord in train_coords:
                values = LatticeEvaluator.compute_uc_dist_Hungary_alg(gen_coord, train_coord)
                dist_to_train.append(values.mean())
            gen_to_train_min_dist.append(np.min(dist_to_train))

        gen_to_train_min_dist = np.asarray(gen_to_train_min_dist, dtype=float)
        metrics_dict = {
            'avg_dist_to_train': float(np.mean(gen_to_train_min_dist)),
            'min_dist_to_train': float(np.min(gen_to_train_min_dist)),
            'median_dist_to_train': float(np.median(gen_to_train_min_dist)),
            'std_dist_to_train': float(np.std(gen_to_train_min_dist)),
            'num_gen': int(len(gen_to_train_min_dist)),
            'num_train': int(len(train_coords)),
        }
        print(metrics_dict)

        if return_per_sample:
            metrics_dict['gen_to_train_min_dist'] = gen_to_train_min_dist.tolist()
        return metrics_dict

    @staticmethod
    def compute_uc_dist_Hungary_alg(coord, gt_coord):
        dist_ij = cdist(coord, gt_coord)
        # row_indices, col_indices = linear_sum_assignment(dist_ij)
        # selected_values = dist_ij[np.arange(col_indices.shape[0]), col_indices]
        selected_values = dist_ij.min(axis=1)
        return selected_values

    @staticmethod
    def compute_uc_dist_greedy_alg(coord, gt_coord):
        matrix = cdist(coord, gt_coord)
        n, m = matrix.shape
        sorted_indices = np.argsort(matrix, axis=1)

        pointers = np.zeros(n, dtype=int)

        used_indices = set()
        result = np.full(n, -1, dtype=int)

        for i in range(n):
            while pointers[i] < m:
                candidate = sorted_indices[i, pointers[i]]
                if candidate not in used_indices:
                    used_indices.add(candidate)
                    result[i] = candidate
                    break
                else:
                    pointers[i] += 1
            if pointers[i] >= m:
                break
        if i < len(result):
            result[i] = np.argmin(matrix[i])
            i += 1
        result_values = matrix[np.arange(result.shape[0]), result]
        return result_values

    @staticmethod
    def compute_cov(coords, gt_coords,
                    struc_cutoff, num_gen_strcuture=None):

        struc_pdist = np.ones((len(coords), len(gt_coords))) * float('inf')
        for i in tqdm(range(len(coords)),desc='Computing Cov'):
            for j in range(len(gt_coords)):
                values = LatticeEvaluator.compute_uc_dist_Hungary_alg(coords[i], gt_coords[j])
                # values = LatticeEvaluator.compute_uc_dist_greedy_alg(coords[i], gt_coords[j])
                struc_pdist[i, j] = values.mean()

        if num_gen_strcuture is None:
            num_gen_crystals = len(coords)

        struc_recall_dist = struc_pdist.min(axis=0)
        struc_precision_dist = struc_pdist.min(axis=1)

        cov_recall = np.sum(
            struc_recall_dist <= struc_cutoff) / len(gt_coords)
        cov_precision = np.sum(
            struc_precision_dist <= struc_cutoff) / num_gen_crystals

        metrics_dict = {
            'cov_recall': cov_recall,
            'cov_precision': cov_precision,
            'amsd_recall': np.mean(struc_recall_dist),
            'amsd_precision': np.mean(struc_precision_dist),
        }

        combined_dist_dict = {
            'struc_recall_dist': struc_recall_dist.tolist(),
            'struc_precision_dist': struc_precision_dist.tolist(),
        }

        return metrics_dict, combined_dist_dict


    def graph_validity(self, coords: List, edges: List, lattice_vectors: List):
        '''

        Args:
            coords:  List(np.ndarray(size=(n,3)))
            lattice_vectors: List(np.ndarray())
            edges:  List(np.ndarray((2, m)))

        Returns:
            periodicity_ratio, mean_symmetry, connectivity_ratio
        '''
        periodicity = []
        connectivity = []
        symmetry_ratio = []
        dangling_node = []
        for i in tqdm(range(len(coords)), desc='Graph validity'):
            lattice_vector = lattice_vectors[i]

            periodicity.append(
                self.is_periodic_necessary_condition(coords[i], lattice_vector.reshape(3, 3), error_bar=self.periodic_error_bar))
            connectivity.append(self.is_connected(edges[i]))
            symmetry_ratio.append(self.central_symmetry(coords[i], error_bar=self.central_symmetry_error_bar))
            dangling_node.append(self.has_dangling_node(coords[i], edges[i]))


        periodicity_ratio = np.array(periodicity).sum() / len(periodicity)
        print(f"Periodicity rate: {periodicity_ratio}")
        mean_symmetry = np.array(symmetry_ratio).sum() / len(periodicity)
        print(f"Mean Central Symmetry rate: {mean_symmetry}")
        connectivity_ratio = np.array(connectivity).sum() / len(connectivity)
        print(f"Connectivity rate: {connectivity_ratio}")
        dangling_node_ratio = 1- np.array(dangling_node).sum() / len(dangling_node)
        print(f'Dangling restriction rate: {dangling_node_ratio}')

        # print(f'Overall validity: ')
        return periodicity_ratio, mean_symmetry, connectivity_ratio, dangling_node_ratio


    @staticmethod
    def has_dangling_node(coords, edge_index):
        if edge_index is None or (edge_index == np.array(None)).any():
            return True
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T

        i, j = edge_index

        degree_dict = {atom_idx: 0 for atom_idx in range(len(coords))}

        for start_node, end_node in zip(i, j):

            degree_dict[start_node] += 1
            degree_dict[end_node] += 1
            

        for degree in degree_dict.values():
            if degree <= 1:
                return True

        return False


    # TODO:  @Wangzhi
    @staticmethod
    def periodical_ratio():
        '''
        TODO:  @Wangzhi
        Returns:

        '''
        pass


    @staticmethod
    def is_periodic_necessary_condition(coords, lattice_vector, error_bar=1e-5):
        # input numpy array
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(lattice_vector, torch.Tensor):
            lattice_vector = lattice_vector.cpu().numpy()

        for d in range(3):
            find_period = False
            for i in range(len(coords)):
                new_coords_i = coords[i] + lattice_vector[d]
                dist = np.abs(new_coords_i - coords)
                if np.any(np.isclose(dist.sum(-1), 0, atol=error_bar)):
                    find_period = True
                    break
            if not find_period:
                return False
        return True

    @staticmethod
    def is_connected(edges):
        if edges is None or (edges == np.array(None)).any():
            return False
        if edges.shape[0] == 2:
            edges = edges.T
        G = nx.Graph()
        G.add_edges_from(edges)
        try:
            res = nx.is_connected(G)
            return res
        except:
            return False
        

    @staticmethod
    def central_symmetry(coords, lattice_vector=None, use_lattice_center=False, error_bar=1e-1):
        '''
        When all nodes are symmetry nodes: 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦   = 1
        When no nodes are symmetry: 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦   = 0
        More #Symmetry nodes, larger 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦  value.
        Less error of symmetry node, larger 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦  value
        None symmetry node position won’t influence the 𝐶𝑒𝑛𝑡𝑟𝑦 𝑠𝑦𝑚𝑚𝑒𝑡𝑟𝑦 value.

        Args:
            coords:
            lattice_vector:
            use_lattice_center:
            error_bar:

        Returns:

        '''
        # if coords1 is symmetry to coords2: coords1 - central == -(coords2 - central)
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if use_lattice_center:
            assert lattice_vector is not None
            center = np.array([[0.,0.,0.]])
            center = np.einsum('bi,bij->bj', center.float(), lattice_vector.unsqueeze(0).float())
        else:
            center = np.array([(coords[:,0].max() + coords[:,0].min()) / 2,
                               (coords[:,1].max() + coords[:,1].min()) / 2,
                               (coords[:,2].max() + coords[:,2].min()) / 2 ])

        dist = coords - center
        dist2 = np.expand_dims(dist, axis=1)
        new_dist = dist + dist2
        descarts_central_dist = np.square(new_dist).sum(axis=-1)**(0.5)
        symmetry_num_per_node = (np.isclose(descarts_central_dist, 0,atol=error_bar)).sum(axis=-1)
        is_symmetry_per_node = symmetry_num_per_node > 0
        symmetry_node_num = is_symmetry_per_node.sum() + 1e-6
        symmetry_node_rate = is_symmetry_per_node.sum() / coords.shape[0]  # Sn

        max_error = max(((dist)**(2)).sum(axis=-1)**(0.5))
        s_error_i = ((new_dist**2).sum(axis=-1)**(0.5)).min(axis=-1)
        s_error_i_ratio = (max_error - s_error_i) / max_error

        

        central_symmetry_rate = symmetry_node_rate * (1 / symmetry_node_num) * (is_symmetry_per_node * s_error_i_ratio).sum()
        return central_symmetry_rate






if __name__ == '__main__':
    ## Example for evaluating generation task
    '''
    Saving lattices:
            np.savez(lattice_name,
                atom_types=gen_atom_types_list[i],
                lengths=gen_lengths_list[i],
                angles=gen_angles_list[i],
                frac_coords=gen_frac_coords_list[i],
                edge_index=edge_index_list[i],
                prop_list=prop_list[i]
                )
    - `cart_coords`: None or cartesian coordinates of each atom, shape `( N, 3)`
    - `frac_coords`: fractional coordinates of each atom, shape `( N, 3)`
    - `atom_types`: atomic number of each atom, shape `( N)`
    - `lengths`: the lengths of the lattice, shape `(3)`
    - `angles`: the angles of the lattice, shape `(3)`
    - `num_atoms`: the number of atoms in each material, shape `(1)`
    - `edge_index`: the edge index of the graph, shape `(2, M)`
    - 'prop_list': (12)


    The following codes will print:
    {'cov_recall': 0.0, 'cov_precision': 0.0, 'amsd_recall': 10.602191118204289, 'amsd_precision': 13.421280170913064}
    Periodicity rate: 0.0
    Mean Central Symmetry rate: 0.5374038704332712
    Connectivity rate: 1.0
    Dangling rate: 0.0
    '''
    dataset = LatticeModulus('/home/grads/jianpengc/datasets/metamaterial/LatticeModulus',file_name='data')
    indices = []
    for i, data in enumerate(tqdm(dataset)):
        if data.num_atoms <= MAX_NODE_NUM and data.num_edges <= MAX_NODE_NUM * 2:
            indices.append(i)
    dataset = dataset[indices]
    split_dict = dataset.get_idx_split(len(dataset), 8000, 1, seed=42)
    train_data = dataset[split_dict['train'].tolist()]
    test_data = dataset[split_dict['test'].tolist()]

    method_eval_paths = {
        'CDVAE': '/home/grads/jianpengc/projects/materials/MetaSymbO/results/cdvae/save_results',
        "LLM-agent": '/home/grads/jianpengc/projects/materials/MetaSymbO/results/pure_llm_prompt',
        'METASYMBO-Mix': '/home/grads/jianpengc/projects/materials/MetaSymbO/results/completely_mix',
        'METASYMBO-Union': '/home/grads/jianpengc/projects/materials/MetaSymbO/results/completely',
    }

    rebuttal_rows = []
    for method_name, eval_path in method_eval_paths.items():
        evaluator = LatticeEvaluator(
            test_datset=test_data[:1000],
            eval_file_path=eval_path,
        )
        dist_metrics = evaluator.eval_distance_to_train(train_dataset=train_data)
        rebuttal_rows.append((
            method_name,
            dist_metrics['avg_dist_to_train'],
            dist_metrics['min_dist_to_train'],
        ))

    print('\n| Method | Avg. dist. to train ↑ | Min. dist. to train ↑ |')
    print('|---|---:|---:|')
    for method_name, avg_dist, min_dist in rebuttal_rows:
        print(f'| {method_name} | {avg_dist:.4f} | {min_dist:.4f} |')

    # evaluator.evaluate_all_uncondition_generation()
    # effectiveness = evaluator.eval_condition_effectiveness()


    # cart_coords: List[np.ndarray] = [d.cart_coords.numpy() for d in dataset]
    # frac_coords: List[np.ndarray] = [d.frac_coords.numpy() for d in dataset]
    # node_types: List[np.ndarray] = [d.node_type.numpy() for d in dataset]
    # edges: List[np.ndarray] = [d.edge_index.numpy() for d in dataset]
    # lattice_vectors: List[np.ndarray] = [d.vector.numpy() for d in dataset]
    #
    # evaluator = LatticeEvaluator(
    #     None,None,
    #     cart_coords,
    #     frac_coords,
    #     node_types,
    #     edges,
    #     lattice_vectors)
    # evaluator.evaluate_all_uncondition_generation()
    # Example for evaluating conditional generation task
    # y_cond = torch.randn((1, 21))  # condition
    # x_gen = torch.randn((15, 3))   # generated lattice conditioned on y_cond

