"""
Symbolic Binding with Role-Filler Composition.

Implements neuro-symbolic binding:
- Symbol-neural representation binding
- Role-filler binding (thematic roles)
- Tensor product composition
- Binding retrieval and unbinding
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class RoleType(Enum):
    """Standard thematic roles for binding."""
    AGENT = "agent"         # Entity performing action
    PATIENT = "patient"     # Entity affected by action
    THEME = "theme"         # Entity moved or changed
    INSTRUMENT = "instrument"  # Tool used
    LOCATION = "location"   # Place
    SOURCE = "source"       # Origin
    GOAL = "goal"           # Destination
    ATTRIBUTE = "attribute" # Property
    VALUE = "value"         # Property value
    RELATION = "relation"   # Relationship type
    ARG1 = "arg1"           # Generic argument 1
    ARG2 = "arg2"           # Generic argument 2
    ARG3 = "arg3"           # Generic argument 3


@dataclass
class RoleBinding:
    """A binding between a role and a filler (value)."""
    role: RoleType
    filler: str  # Symbol name
    neural_rep: np.ndarray  # Neural representation of filler
    confidence: float = 1.0


@dataclass
class CompositeBinding:
    """
    A composite binding representing a structured concept.

    Example: "chase(dog, cat)" binds:
    - symbol: "chase"
    - neural_rep: embedding of chase concept
    - role_bindings: {AGENT: dog, PATIENT: cat}
    """
    symbol: str
    neural_rep: np.ndarray
    role_bindings: Dict[RoleType, RoleBinding] = field(default_factory=dict)
    constraints: List[Callable[[Dict], bool]] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_filler(self, role: RoleType) -> Optional[str]:
        """Get the filler for a given role."""
        binding = self.role_bindings.get(role)
        return binding.filler if binding else None

    def get_neural_rep(self, role: RoleType) -> Optional[np.ndarray]:
        """Get neural representation for a role's filler."""
        binding = self.role_bindings.get(role)
        return binding.neural_rep if binding else None

    def check_constraints(self, context: Dict) -> bool:
        """Check if all constraints are satisfied."""
        return all(c(context) for c in self.constraints)

    def roles(self) -> Set[RoleType]:
        """Get all bound roles."""
        return set(self.role_bindings.keys())


class HRROperations:
    """
    Holographic Reduced Representations (HRR) operations.

    Implements circular convolution for compositional binding.
    """

    @staticmethod
    def circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular convolution for binding.

        a * b where * is circular convolution (implemented via FFT)
        """
        return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))

    @staticmethod
    def circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Circular correlation for unbinding.

        a # b retrieves what was bound with a
        """
        return np.real(np.fft.ifft(np.conj(np.fft.fft(a)) * np.fft.fft(b)))

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return v
        return v / norm


class TPROperations:
    """
    Tensor Product Representation (TPR) operations.

    Implements tensor product for structured binding.
    """

    @staticmethod
    def tensor_product(role: np.ndarray, filler: np.ndarray) -> np.ndarray:
        """
        Compute tensor product of role and filler.

        Returns flattened outer product.
        """
        return np.outer(role, filler).flatten()

    @staticmethod
    def sum_tpr(bindings: List[np.ndarray]) -> np.ndarray:
        """Sum multiple TPR bindings into a single representation."""
        if not bindings:
            return np.array([])
        return np.sum(bindings, axis=0)

    @staticmethod
    def unbind(tpr: np.ndarray, role: np.ndarray, filler_dim: int) -> np.ndarray:
        """
        Unbind role from TPR to retrieve filler.

        Approximate unbinding via matrix multiplication.
        """
        role_dim = len(role)
        tpr_matrix = tpr.reshape(role_dim, filler_dim)
        return tpr_matrix.T @ role


class SymbolicBinder:
    """
    Bind symbols to neural representations with role-filler structure.

    Supports:
    - Simple symbol-neural binding
    - Role-filler structured binding
    - Tensor product composition
    - HRR-based composition
    - Binding retrieval and unbinding
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        role_dim: int = 64,
        binding_method: str = "tpr",  # "tpr" or "hrr"
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.role_dim = role_dim
        self.binding_method = binding_method
        self._rng = np.random.default_rng(random_seed)

        # Symbol registry
        self._symbols: Dict[str, np.ndarray] = {}

        # Role vectors
        self._role_vectors: Dict[RoleType, np.ndarray] = {}
        self._initialize_roles()

        # Binding history
        self._bindings: Dict[str, CompositeBinding] = {}

        # Statistics
        self._n_bindings = 0
        self._n_retrievals = 0

    def _initialize_roles(self) -> None:
        """Initialize random orthogonal role vectors."""
        # Create orthogonal role vectors using QR decomposition
        n_roles = len(RoleType)
        random_matrix = self._rng.normal(0, 1, (self.role_dim, n_roles))
        q, _ = np.linalg.qr(random_matrix)

        for i, role in enumerate(RoleType):
            if i < q.shape[1]:
                self._role_vectors[role] = q[:, i]
            else:
                self._role_vectors[role] = HRROperations.normalize(
                    self._rng.normal(0, 1, self.role_dim)
                )

    def register_symbol(
        self,
        symbol: str,
        neural_rep: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Register a symbol with its neural representation.

        If no representation is provided, creates a random one.
        """
        if neural_rep is None:
            neural_rep = HRROperations.normalize(
                self._rng.normal(0, 1, self.embedding_dim)
            )
        else:
            neural_rep = HRROperations.normalize(neural_rep)

        self._symbols[symbol] = neural_rep
        return neural_rep

    def get_symbol(self, symbol: str) -> Optional[np.ndarray]:
        """Get neural representation for a symbol."""
        return self._symbols.get(symbol)

    def bind(
        self,
        symbol: str,
        neural_rep: Optional[np.ndarray] = None,
        roles: Optional[Dict[RoleType, Tuple[str, np.ndarray]]] = None,
    ) -> CompositeBinding:
        """
        Create a composite binding.

        Args:
            symbol: The main symbol (e.g., "chase")
            neural_rep: Neural representation (optional)
            roles: Dict mapping RoleType to (filler_name, filler_rep) tuples
        """
        self._n_bindings += 1

        # Get or create neural rep for main symbol
        if neural_rep is None:
            if symbol in self._symbols:
                neural_rep = self._symbols[symbol]
            else:
                neural_rep = self.register_symbol(symbol)
        else:
            neural_rep = HRROperations.normalize(neural_rep)
            self._symbols[symbol] = neural_rep

        # Create role bindings
        role_bindings = {}
        if roles:
            for role, (filler, filler_rep) in roles.items():
                if filler_rep is None:
                    if filler in self._symbols:
                        filler_rep = self._symbols[filler]
                    else:
                        filler_rep = self.register_symbol(filler)

                role_bindings[role] = RoleBinding(
                    role=role,
                    filler=filler,
                    neural_rep=HRROperations.normalize(filler_rep),
                )

        binding = CompositeBinding(
            symbol=symbol,
            neural_rep=neural_rep,
            role_bindings=role_bindings,
        )

        self._bindings[symbol] = binding
        return binding

    def bind_structure(
        self,
        predicate: str,
        args: Dict[str, Tuple[str, Optional[np.ndarray]]],
        role_mapping: Optional[Dict[str, RoleType]] = None,
    ) -> CompositeBinding:
        """
        Bind a predicate with named arguments.

        Args:
            predicate: The predicate symbol (e.g., "loves")
            args: Dict of argument names to (symbol, neural_rep) tuples
            role_mapping: Optional mapping from arg names to RoleTypes
        """
        # Default role mapping
        if role_mapping is None:
            role_mapping = {
                "arg1": RoleType.ARG1,
                "arg2": RoleType.ARG2,
                "arg3": RoleType.ARG3,
                "agent": RoleType.AGENT,
                "patient": RoleType.PATIENT,
                "theme": RoleType.THEME,
            }

        roles = {}
        for arg_name, (symbol, neural_rep) in args.items():
            role = role_mapping.get(arg_name, RoleType.ARG1)
            roles[role] = (symbol, neural_rep)

        return self.bind(predicate, roles=roles)

    def compose(
        self,
        bindings: List[CompositeBinding],
    ) -> np.ndarray:
        """
        Compose multiple bindings into a single representation.

        Uses either TPR or HRR based on binding_method.
        """
        if not bindings:
            return np.zeros(self.embedding_dim)

        if self.binding_method == "tpr":
            return self._compose_tpr(bindings)
        else:
            return self._compose_hrr(bindings)

    def _compose_tpr(self, bindings: List[CompositeBinding]) -> np.ndarray:
        """Compose using Tensor Product Representations."""
        composed_parts = []

        for binding in bindings:
            # Add main symbol representation
            composed_parts.append(binding.neural_rep)

            # Add role-filler bindings
            for role, role_binding in binding.role_bindings.items():
                role_vec = self._role_vectors[role]
                # Use circular convolution as approximation to full TPR
                bound = HRROperations.circular_convolution(
                    np.tile(role_vec, self.embedding_dim // self.role_dim + 1)[:self.embedding_dim],
                    role_binding.neural_rep
                )
                composed_parts.append(bound)

        # Sum and normalize
        result = np.sum(composed_parts, axis=0)
        return HRROperations.normalize(result)

    def _compose_hrr(self, bindings: List[CompositeBinding]) -> np.ndarray:
        """Compose using Holographic Reduced Representations."""
        composed = np.zeros(self.embedding_dim)

        for binding in bindings:
            # Convolve symbol with each role-filler pair
            trace = binding.neural_rep.copy()

            for role, role_binding in binding.role_bindings.items():
                role_vec = self._role_vectors[role]
                role_vec_expanded = np.tile(role_vec, self.embedding_dim // self.role_dim + 1)[:self.embedding_dim]

                # Bind role to filler
                bound = HRROperations.circular_convolution(
                    role_vec_expanded,
                    role_binding.neural_rep
                )
                trace += bound

            composed += trace

        return HRROperations.normalize(composed)

    def retrieve_by_role(
        self,
        composed: np.ndarray,
        role: RoleType,
    ) -> np.ndarray:
        """
        Retrieve filler for a given role from composed representation.

        Uses unbinding operation.
        """
        self._n_retrievals += 1

        role_vec = self._role_vectors[role]
        role_vec_expanded = np.tile(role_vec, self.embedding_dim // self.role_dim + 1)[:self.embedding_dim]

        # Unbind using circular correlation
        retrieved = HRROperations.circular_correlation(role_vec_expanded, composed)
        return HRROperations.normalize(retrieved)

    def retrieve_nearest_symbol(
        self,
        representation: np.ndarray,
        top_k: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Find the nearest registered symbol(s) to a representation.
        """
        similarities = []

        for symbol, rep in self._symbols.items():
            sim = float(np.dot(representation, rep))
            similarities.append((symbol, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def unbind(
        self,
        binding: CompositeBinding,
    ) -> Dict[RoleType, Tuple[str, float]]:
        """
        Unbind a composite binding to retrieve role-filler pairs.

        Returns dict of role -> (nearest_symbol, similarity)
        """
        results = {}

        composed = self.compose([binding])

        for role in binding.roles():
            retrieved = self.retrieve_by_role(composed, role)
            matches = self.retrieve_nearest_symbol(retrieved, top_k=1)

            if matches:
                results[role] = matches[0]

        return results

    def similarity(
        self,
        binding1: CompositeBinding,
        binding2: CompositeBinding,
    ) -> float:
        """Compute similarity between two bindings."""
        rep1 = self.compose([binding1])
        rep2 = self.compose([binding2])
        return float(np.dot(rep1, rep2))

    def analogical_bind(
        self,
        source_binding: CompositeBinding,
        target_mapping: Dict[str, str],
    ) -> CompositeBinding:
        """
        Create analogical binding by mapping source elements to target.

        Args:
            source_binding: Original binding
            target_mapping: Dict mapping source symbols to target symbols
        """
        # Map main symbol
        new_symbol = target_mapping.get(source_binding.symbol, source_binding.symbol)

        # Map role fillers
        new_roles = {}
        for role, role_binding in source_binding.role_bindings.items():
            new_filler = target_mapping.get(role_binding.filler, role_binding.filler)
            new_rep = self._symbols.get(new_filler)
            if new_rep is None:
                new_rep = self.register_symbol(new_filler)
            new_roles[role] = (new_filler, new_rep)

        return self.bind(new_symbol, roles=new_roles)

    def verify_binding_consistency(
        self,
        binding: CompositeBinding,
    ) -> Tuple[bool, float]:
        """
        Verify that a binding can be recovered through unbinding.

        Returns (is_consistent, avg_similarity)
        """
        composed = self.compose([binding])
        similarities = []

        for role, role_binding in binding.role_bindings.items():
            retrieved = self.retrieve_by_role(composed, role)
            sim = float(np.dot(retrieved, role_binding.neural_rep))
            similarities.append(sim)

        if not similarities:
            return True, 1.0

        avg_sim = np.mean(similarities)
        is_consistent = avg_sim > 0.3  # Threshold

        return is_consistent, float(avg_sim)

    def get_role_vector(self, role: RoleType) -> np.ndarray:
        """Get the role vector for a given role type."""
        return self._role_vectors[role]

    def list_symbols(self) -> List[str]:
        """List all registered symbols."""
        return list(self._symbols.keys())

    def statistics(self) -> Dict[str, Any]:
        """Get binder statistics."""
        return {
            "n_symbols": len(self._symbols),
            "n_bindings": self._n_bindings,
            "n_retrievals": self._n_retrievals,
            "embedding_dim": self.embedding_dim,
            "role_dim": self.role_dim,
            "binding_method": self.binding_method,
            "n_roles": len(self._role_vectors),
        }
