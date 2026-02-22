"""
Causal Structure Discovery.

Implements algorithms for learning causal graph structure from data:
- PC Algorithm (constraint-based)
- Conditional independence testing
- Edge orientation rules
- Uncertainty quantification
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np
from itertools import combinations
from scipy import stats


class EdgeType(Enum):
    """Types of edges in a causal graph."""

    UNDIRECTED = "undirected"  # X -- Y
    DIRECTED = "directed"  # X -> Y
    BIDIRECTED = "bidirected"  # X <-> Y (latent confounder)


@dataclass
class CausalEdge:
    """An edge in a causal graph."""

    source: str
    target: str
    edge_type: EdgeType = EdgeType.UNDIRECTED
    confidence: float = 1.0
    p_value: Optional[float] = None


@dataclass
class CausalGraph:
    """A causal graph (DAG or CPDAG)."""

    nodes: Set[str]
    edges: List[CausalEdge]
    is_dag: bool = False

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return adjacency matrix and node list."""
        node_list = sorted(self.nodes)
        n = len(node_list)
        node_idx = {node: i for i, node in enumerate(node_list)}

        adj = np.zeros((n, n))
        for edge in self.edges:
            i, j = node_idx[edge.source], node_idx[edge.target]
            if edge.edge_type == EdgeType.UNDIRECTED:
                adj[i, j] = adj[j, i] = 1
            elif edge.edge_type == EdgeType.DIRECTED:
                adj[i, j] = 1
            elif edge.edge_type == EdgeType.BIDIRECTED:
                adj[i, j] = adj[j, i] = 2

        return adj, node_list

    def parents(self, node: str) -> Set[str]:
        """Get parents of a node."""
        return {
            e.source for e in self.edges if e.target == node and e.edge_type == EdgeType.DIRECTED
        }

    def children(self, node: str) -> Set[str]:
        """Get children of a node."""
        return {
            e.target for e in self.edges if e.source == node and e.edge_type == EdgeType.DIRECTED
        }

    def neighbors(self, node: str) -> Set[str]:
        """Get all neighbors (adjacent nodes)."""
        neighbors = set()
        for e in self.edges:
            if e.source == node:
                neighbors.add(e.target)
            elif e.target == node:
                neighbors.add(e.source)
        return neighbors

    def get_edge(self, source: str, target: str) -> Optional[CausalEdge]:
        """Get edge between two nodes if it exists."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
            if e.edge_type == EdgeType.UNDIRECTED:
                if e.source == target and e.target == source:
                    return e
        return None


class ConditionalIndependenceTest:
    """
    Statistical tests for conditional independence.

    Implements multiple testing methods for X _|_ Y | Z.
    """

    def __init__(
        self,
        method: str = "partial_correlation",
        alpha: float = 0.01,
    ):
        self.method = method
        self.alpha = alpha

        self._n_tests = 0

    def test(
        self,
        data: np.ndarray,
        x_idx: int,
        y_idx: int,
        z_idx: Optional[List[int]] = None,
        var_names: Optional[List[str]] = None,
    ) -> Tuple[bool, float, float]:
        """
        Test if X _|_ Y | Z.

        Args:
            data: (n_samples, n_variables) data matrix
            x_idx: Index of X variable
            y_idx: Index of Y variable
            z_idx: Indices of conditioning set Z

        Returns:
            (is_independent, test_statistic, p_value)
        """
        self._n_tests += 1
        z_idx = z_idx or []

        if self.method == "partial_correlation":
            return self._partial_correlation_test(data, x_idx, y_idx, z_idx)
        elif self.method == "mutual_information":
            return self._mutual_info_test(data, x_idx, y_idx, z_idx)
        else:
            return self._partial_correlation_test(data, x_idx, y_idx, z_idx)

    def _partial_correlation_test(
        self,
        data: np.ndarray,
        x_idx: int,
        y_idx: int,
        z_idx: List[int],
    ) -> Tuple[bool, float, float]:
        """Test using partial correlation."""
        n_samples = data.shape[0]

        if len(z_idx) == 0:
            # Simple correlation
            r = np.corrcoef(data[:, x_idx], data[:, y_idx])[0, 1]
        else:
            # Partial correlation via regression residuals
            X = data[:, x_idx]
            Y = data[:, y_idx]
            Z = data[:, z_idx]

            # Regress X on Z
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            Z_pinv = np.linalg.pinv(Z)
            X_resid = X - Z @ (Z_pinv @ X)
            Y_resid = Y - Z @ (Z_pinv @ Y)

            # Correlation of residuals
            r = np.corrcoef(X_resid, Y_resid)[0, 1]

        # Handle numerical issues
        if np.isnan(r):
            r = 0.0

        # Fisher's z-transformation for p-value
        n_conditioning = len(z_idx)
        dof = n_samples - n_conditioning - 2

        if dof <= 0 or abs(r) >= 1.0:
            return True, 0.0, 1.0

        z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        is_independent = p_value > self.alpha

        return is_independent, float(r), float(p_value)

    def _mutual_info_test(
        self,
        data: np.ndarray,
        x_idx: int,
        y_idx: int,
        z_idx: List[int],
    ) -> Tuple[bool, float, float]:
        """Test using mutual information (discretized)."""
        n_samples = data.shape[0]
        n_bins = max(5, int(np.sqrt(n_samples / 5)))

        X = data[:, x_idx]
        Y = data[:, y_idx]

        # Discretize
        X_disc = np.digitize(X, np.histogram_bin_edges(X, bins=n_bins))
        Y_disc = np.digitize(Y, np.histogram_bin_edges(Y, bins=n_bins))

        if len(z_idx) == 0:
            # Unconditional MI
            mi = self._compute_mi(X_disc, Y_disc)
        else:
            # Conditional MI (averaged over Z)
            Z = data[:, z_idx]
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            Z_disc = np.apply_along_axis(
                lambda col: np.digitize(col, np.histogram_bin_edges(col, bins=n_bins)), 0, Z
            )
            mi = self._compute_cmi(X_disc, Y_disc, Z_disc)

        # Chi-squared approximation for p-value
        chi2 = 2 * n_samples * mi
        dof = (n_bins - 1) ** 2
        p_value = 1 - stats.chi2.cdf(chi2, dof)

        is_independent = p_value > self.alpha

        return is_independent, float(mi), float(p_value)

    def _compute_mi(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute mutual information."""
        # Joint distribution
        joint_counts = {}
        for x, y in zip(X, Y):
            key = (x, y)
            joint_counts[key] = joint_counts.get(key, 0) + 1

        n = len(X)
        mi = 0.0

        # Marginals
        px = np.bincount(X.astype(int)) / n
        py = np.bincount(Y.astype(int)) / n

        for (x, y), count in joint_counts.items():
            pxy = count / n
            if pxy > 0 and px[int(x)] > 0 and py[int(y)] > 0:
                mi += pxy * np.log(pxy / (px[int(x)] * py[int(y)]))

        return max(0.0, mi)

    def _compute_cmi(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> float:
        """Compute conditional mutual information."""
        # Simplified: average MI over Z strata
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)

        # Create Z groups
        z_keys = [tuple(row) for row in Z]
        unique_z = list(set(z_keys))

        cmi = 0.0
        total_weight = 0.0

        for z_val in unique_z:
            mask = [z == z_val for z in z_keys]
            mask = np.array(mask)
            if mask.sum() < 5:
                continue

            X_z = X[mask]
            Y_z = Y[mask]

            weight = mask.sum() / len(X)
            cmi += weight * self._compute_mi(X_z, Y_z)
            total_weight += weight

        if total_weight > 0:
            cmi /= total_weight

        return cmi

    def statistics(self) -> Dict[str, Any]:
        """Get test statistics."""
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self._n_tests,
        }


class CausalDiscovery:
    """
    Causal structure learning using constraint-based methods.

    Implements the PC algorithm:
    1. Start with complete undirected graph
    2. Remove edges based on conditional independence tests
    3. Orient edges using d-separation rules
    """

    def __init__(
        self,
        alpha: float = 0.01,
        max_conditioning_set: int = 5,
        ci_test: Optional[ConditionalIndependenceTest] = None,
    ):
        self.alpha = alpha
        self.max_conditioning_set = max_conditioning_set
        self.ci_test = ci_test or ConditionalIndependenceTest(alpha=alpha)

        # Separation sets (for orientation)
        self._sep_sets: Dict[Tuple[str, str], Set[str]] = {}

        # Edge confidence scores
        self._edge_confidences: Dict[Tuple[str, str], float] = {}

        # Statistics
        self._n_discoveries = 0

    def pc_algorithm(
        self,
        data: np.ndarray,
        var_names: Optional[List[str]] = None,
    ) -> CausalGraph:
        """
        Run PC algorithm to discover causal structure.

        Args:
            data: (n_samples, n_variables) data matrix
            var_names: Optional variable names

        Returns:
            CausalGraph (CPDAG - may have undirected edges)
        """
        self._n_discoveries += 1

        n_vars = data.shape[1]
        if var_names is None:
            var_names = [f"V{i}" for i in range(n_vars)]

        # Step 1: Start with complete undirected graph
        skeleton = self._learn_skeleton(data, var_names)

        # Step 2: Orient edges
        cpdag = self._orient_edges(skeleton, var_names)

        return cpdag

    def _learn_skeleton(
        self,
        data: np.ndarray,
        var_names: List[str],
    ) -> CausalGraph:
        """Learn the skeleton (undirected graph) using CI tests."""
        n_vars = len(var_names)
        {name: i for i, name in enumerate(var_names)}

        # Start with complete graph
        adj = np.ones((n_vars, n_vars)) - np.eye(n_vars)

        self._sep_sets = {}
        self._edge_confidences = {}

        # Test edges with increasing conditioning set size
        for cond_size in range(self.max_conditioning_set + 1):
            # List edges to test
            edges_to_test = []
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if adj[i, j] > 0:
                        edges_to_test.append((i, j))

            for i, j in edges_to_test:
                if adj[i, j] == 0:
                    continue

                # Get potential conditioning sets from neighbors
                neighbors_i = set(np.where(adj[i] > 0)[0]) - {j}
                neighbors_j = set(np.where(adj[j] > 0)[0]) - {i}
                potential_conds = neighbors_i | neighbors_j

                # Test all conditioning sets of current size
                for cond_set in combinations(potential_conds, cond_size):
                    cond_list = list(cond_set)
                    is_ind, stat, p_val = self.ci_test.test(data, i, j, cond_list)

                    if is_ind:
                        # Remove edge
                        adj[i, j] = adj[j, i] = 0

                        # Record separation set
                        key = (var_names[i], var_names[j])
                        self._sep_sets[key] = {var_names[k] for k in cond_list}
                        self._sep_sets[(var_names[j], var_names[i])] = self._sep_sets[key]

                        break

                # Record confidence
                if adj[i, j] > 0:
                    key = (var_names[i], var_names[j])
                    self._edge_confidences[key] = 1.0 - self.ci_test.alpha

        # Build graph
        nodes = set(var_names)
        edges = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] > 0:
                    conf = self._edge_confidences.get((var_names[i], var_names[j]), 1.0)
                    edges.append(
                        CausalEdge(
                            source=var_names[i],
                            target=var_names[j],
                            edge_type=EdgeType.UNDIRECTED,
                            confidence=conf,
                        )
                    )

        return CausalGraph(nodes=nodes, edges=edges, is_dag=False)

    def _orient_edges(
        self,
        skeleton: CausalGraph,
        var_names: List[str],
    ) -> CausalGraph:
        """Orient edges to form CPDAG."""
        # Make copy of edges
        oriented_edges = {(e.source, e.target): e.edge_type for e in skeleton.edges}

        # Also add reverse for undirected
        for e in skeleton.edges:
            if e.edge_type == EdgeType.UNDIRECTED:
                oriented_edges[(e.target, e.source)] = EdgeType.UNDIRECTED

        # Rule 1: Orient v-structures (X -> Z <- Y)
        # If X -- Z -- Y and X not adjacent to Y and Z not in sep(X,Y)
        for z in skeleton.nodes:
            neighbors = skeleton.neighbors(z)
            for x, y in combinations(neighbors, 2):
                # Check if X -- Y
                if skeleton.get_edge(x, y) is not None:
                    continue

                # Check separation set
                sep = self._sep_sets.get((x, y), set())
                if z not in sep:
                    # Orient as v-structure: X -> Z <- Y
                    oriented_edges[(x, z)] = EdgeType.DIRECTED
                    if (z, x) in oriented_edges:
                        del oriented_edges[(z, x)]

                    oriented_edges[(y, z)] = EdgeType.DIRECTED
                    if (z, y) in oriented_edges:
                        del oriented_edges[(z, y)]

        # Apply orientation propagation rules
        changed = True
        while changed:
            changed = False

            for x, z in list(oriented_edges.keys()):
                if oriented_edges[(x, z)] != EdgeType.DIRECTED:
                    continue

                # Rule 2: Orient Z -- Y as Z -> Y if X -> Z
                for y in skeleton.neighbors(z):
                    if y == x:
                        continue
                    if (z, y) in oriented_edges and oriented_edges[(z, y)] == EdgeType.UNDIRECTED:
                        if skeleton.get_edge(x, y) is None:
                            oriented_edges[(z, y)] = EdgeType.DIRECTED
                            if (y, z) in oriented_edges:
                                del oriented_edges[(y, z)]
                            changed = True

        # Build final graph
        final_edges = []
        seen = set()
        for (src, tgt), etype in oriented_edges.items():
            if etype == EdgeType.UNDIRECTED:
                # Only add once
                key = tuple(sorted([src, tgt]))
                if key in seen:
                    continue
                seen.add(key)
                final_edges.append(
                    CausalEdge(
                        source=src,
                        target=tgt,
                        edge_type=EdgeType.UNDIRECTED,
                        confidence=self._edge_confidences.get((src, tgt), 1.0),
                    )
                )
            else:
                final_edges.append(
                    CausalEdge(
                        source=src,
                        target=tgt,
                        edge_type=etype,
                        confidence=self._edge_confidences.get((src, tgt), 1.0),
                    )
                )

        return CausalGraph(nodes=skeleton.nodes, edges=final_edges, is_dag=False)

    def estimate_structure_uncertainty(
        self,
        data: np.ndarray,
        var_names: Optional[List[str]] = None,
        n_bootstrap: int = 100,
    ) -> Dict[Tuple[str, str], float]:
        """
        Estimate uncertainty in discovered edges using bootstrap.

        Returns confidence (frequency of edge appearing) for each edge.
        """
        n_samples, n_vars = data.shape
        if var_names is None:
            var_names = [f"V{i}" for i in range(n_vars)]

        edge_counts: Dict[Tuple[str, str], int] = {}

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_data = data[indices]

            # Discover structure
            graph = self.pc_algorithm(boot_data, var_names)

            # Count edges
            for edge in graph.edges:
                key = tuple(sorted([edge.source, edge.target]))
                edge_counts[key] = edge_counts.get(key, 0) + 1

        # Compute frequencies
        edge_confidence = {key: count / n_bootstrap for key, count in edge_counts.items()}

        return edge_confidence

    def compare_structures(
        self,
        graph1: CausalGraph,
        graph2: CausalGraph,
    ) -> Dict[str, Any]:
        """
        Compare two causal graphs.

        Returns structural Hamming distance and other metrics.
        """

        # Get edge sets (as undirected for skeleton comparison)
        def edge_set(g: CausalGraph) -> Set[Tuple[str, str]]:
            edges = set()
            for e in g.edges:
                key = tuple(sorted([e.source, e.target]))
                edges.add(key)
            return edges

        edges1 = edge_set(graph1)
        edges2 = edge_set(graph2)

        # Structural Hamming Distance for skeleton
        skeleton_shd = len(edges1 ^ edges2)

        # Edge precision/recall
        true_positives = len(edges1 & edges2)
        precision = true_positives / len(edges2) if edges2 else 0.0
        recall = true_positives / len(edges1) if edges1 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "skeleton_shd": skeleton_shd,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "edges_graph1": len(edges1),
            "edges_graph2": len(edges2),
            "common_edges": true_positives,
        }

    def score_graph(
        self,
        graph: CausalGraph,
        data: np.ndarray,
        var_names: List[str],
    ) -> float:
        """
        Score a causal graph using BIC.

        Lower score is better.
        """
        n_samples = data.shape[0]
        var_idx = {name: i for i, name in enumerate(var_names)}

        total_score = 0.0

        for node in graph.nodes:
            # Get parents
            parents = graph.parents(node)
            node_idx = var_idx[node]

            # Regress node on parents
            Y = data[:, node_idx]

            if parents:
                parent_indices = [var_idx[p] for p in parents]
                X = data[:, parent_indices]
                if X.ndim == 1:
                    X = X.reshape(-1, 1)

                # Fit linear regression
                X_pinv = np.linalg.pinv(X)
                coef = X_pinv @ Y
                Y_pred = X @ coef
                residuals = Y - Y_pred
            else:
                residuals = Y - Y.mean()

            # Compute log-likelihood
            rss = np.sum(residuals**2)
            var = rss / n_samples
            ll = -0.5 * n_samples * (np.log(2 * np.pi * var) + 1)

            # BIC penalty
            n_params = len(parents) + 1  # coefficients + intercept
            bic = -2 * ll + n_params * np.log(n_samples)

            total_score += bic

        return total_score

    def statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        return {
            "n_discoveries": self._n_discoveries,
            "alpha": self.alpha,
            "max_conditioning_set": self.max_conditioning_set,
            "ci_test_stats": self.ci_test.statistics(),
        }
