"""
Attention Competition: Mechanism for selecting content to enter the workspace.

Implements an attention-based competition where proposals from different
cognitive modules compete for access to the limited-capacity global workspace.
Winners are selected based on activation strength, confidence, and relevance.

Based on:
- Biased Competition Model (Desimone & Duncan, 1995)
- Global Workspace competition dynamics
- arXiv:2103.01197 - Bandwidth limits force competition
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    from .module_interface import ModuleProposal, ModuleType, ContentType
except ImportError:
    from module_interface import ModuleProposal, ModuleType, ContentType


@dataclass
class CompetitionParams:
    """Parameters for attention competition."""
    n_features: int = 64
    capacity: int = 7  # Maximum winners (workspace capacity)
    temperature: float = 1.0  # Softmax temperature for selection
    top_down_weight: float = 0.3  # Weight for top-down attention
    bottom_up_weight: float = 0.7  # Weight for bottom-up salience
    inhibition_strength: float = 0.2  # Lateral inhibition between proposals
    recurrent_iterations: int = 3  # Iterations for winner-take-all dynamics


@dataclass
class CompetitionResult:
    """Result of attention competition."""
    winners: List[ModuleProposal]
    all_scores: List[Tuple[ModuleProposal, float]]
    max_score: float
    competition_entropy: float
    iterations_used: int


class AttentionCompetition:
    """
    Attention-based competition mechanism for workspace access.

    Proposals compete based on multiple factors:
    1. **Activation**: Raw activation strength of the proposal
    2. **Confidence**: Module's confidence in the proposal
    3. **Relevance**: Relevance to current context/goals
    4. **Top-down bias**: Alignment with current attention focus
    5. **Lateral inhibition**: Suppression from competing proposals

    The competition uses recurrent dynamics where proposals inhibit
    each other until a stable state is reached (winner-take-all).
    """

    def __init__(self, params: Optional[CompetitionParams] = None):
        self.params = params or CompetitionParams()

        # Top-down attention bias (what to attend to)
        self._attention_bias: np.ndarray = np.zeros(self.params.n_features)

        # Context vector for relevance computation
        self._context: np.ndarray = np.zeros(self.params.n_features)

        # Priority weights for different content types
        self._content_priorities: Dict[ContentType, float] = {
            ContentType.ERROR: 1.5,        # Prediction errors are important
            ContentType.EMOTION: 1.3,      # Emotions get priority
            ContentType.INTENTION: 1.2,    # Goals are important
            ContentType.QUERY: 1.1,        # Information requests
            ContentType.PERCEPT: 1.0,      # Standard sensory
            ContentType.BELIEF: 1.0,       # Beliefs
            ContentType.MEMORY: 0.9,       # Retrieved memories
            ContentType.PREDICTION: 0.9,   # Predictions
            ContentType.ACTION: 0.8,       # Motor commands
            ContentType.RESPONSE: 0.8,     # Responses
            ContentType.METACOGNITIVE: 1.4,  # Self-reflection is important
        }

    def compete(self, proposals: List[ModuleProposal]) -> CompetitionResult:
        """
        Run competition among proposals.

        Args:
            proposals: List of proposals from all modules

        Returns:
            CompetitionResult with winners and scores
        """
        if not proposals:
            return CompetitionResult(
                winners=[],
                all_scores=[],
                max_score=0.0,
                competition_entropy=0.0,
                iterations_used=0,
            )

        # Compute initial scores
        scores = np.array([self._compute_score(p) for p in proposals])

        # Recurrent competition (winner-take-all dynamics)
        iterations = 0
        for _ in range(self.params.recurrent_iterations):
            iterations += 1

            # Apply lateral inhibition
            new_scores = self._apply_inhibition(proposals, scores)

            # Check for convergence
            if np.allclose(scores, new_scores, rtol=1e-3):
                break

            scores = new_scores

        # Apply softmax for probabilistic selection
        probs = self._softmax(scores)

        # Select winners (top-k by probability)
        n_winners = min(self.params.capacity, len(proposals))
        winner_indices = np.argsort(probs)[-n_winners:][::-1]

        winners = [proposals[i] for i in winner_indices]
        all_scores = [(proposals[i], float(scores[i])) for i in range(len(proposals))]

        # Compute competition entropy (measure of competition uncertainty)
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        return CompetitionResult(
            winners=winners,
            all_scores=all_scores,
            max_score=float(np.max(scores)),
            competition_entropy=float(entropy),
            iterations_used=iterations,
        )

    def _compute_score(self, proposal: ModuleProposal) -> float:
        """
        Compute competition score for a proposal.

        Score = (bottom_up_salience * w_bu + top_down_bias * w_td) * type_priority
        """
        # Bottom-up salience (based on proposal properties)
        bottom_up = (
            proposal.activation * 0.4 +
            proposal.confidence * 0.3 +
            proposal.relevance * 0.3
        )

        # Top-down attention bias
        top_down = self._compute_top_down_match(proposal)

        # Content type priority
        type_priority = self._content_priorities.get(proposal.content_type, 1.0)

        # Combined score
        score = (
            self.params.bottom_up_weight * bottom_up +
            self.params.top_down_weight * top_down
        ) * type_priority

        return float(np.clip(score, 0, 2))

    def _compute_top_down_match(self, proposal: ModuleProposal) -> float:
        """Compute match between proposal and current attention bias."""
        if np.linalg.norm(self._attention_bias) < 1e-8:
            return 0.5  # No bias set

        # Resize content if needed
        content = proposal.content
        if len(content) != len(self._attention_bias):
            content = np.resize(content, len(self._attention_bias))

        # Cosine similarity
        norm_content = np.linalg.norm(content)
        norm_bias = np.linalg.norm(self._attention_bias)

        if norm_content < 1e-8:
            return 0.0

        similarity = np.dot(content, self._attention_bias) / (norm_content * norm_bias)
        return float(np.clip((similarity + 1) / 2, 0, 1))

    def _apply_inhibition(
        self,
        proposals: List[ModuleProposal],
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Apply lateral inhibition between competing proposals.

        Strong proposals inhibit weaker ones, implementing
        winner-take-all dynamics.
        """
        n = len(proposals)
        new_scores = scores.copy()

        for i in range(n):
            for j in range(n):
                if i != j:
                    # Compute inhibition based on content similarity
                    similarity = self._content_similarity(
                        proposals[i].content,
                        proposals[j].content,
                    )

                    # Stronger proposals inhibit weaker ones
                    if scores[j] > scores[i]:
                        inhibition = self.params.inhibition_strength * similarity * scores[j]
                        new_scores[i] = max(0, new_scores[i] - inhibition)

        # Normalize to maintain total activation
        if np.sum(new_scores) > 0:
            new_scores = new_scores * (np.sum(scores) / np.sum(new_scores))

        return new_scores

    def _content_similarity(self, content1: np.ndarray, content2: np.ndarray) -> float:
        """Compute similarity between two content vectors."""
        # Handle size mismatch
        if len(content1) != len(content2):
            min_len = min(len(content1), len(content2))
            content1 = content1[:min_len]
            content2 = content2[:min_len]

        norm1 = np.linalg.norm(content1)
        norm2 = np.linalg.norm(content2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        similarity = np.dot(content1, content2) / (norm1 * norm2)
        return float(np.clip((similarity + 1) / 2, 0, 1))

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Apply softmax with temperature."""
        scaled = scores / self.params.temperature
        exp_scores = np.exp(scaled - np.max(scaled))  # Numerical stability
        return exp_scores / np.sum(exp_scores)

    def set_attention_bias(self, bias: np.ndarray) -> None:
        """Set top-down attention bias."""
        if len(bias) != self.params.n_features:
            bias = np.resize(bias, self.params.n_features)
        self._attention_bias = bias / (np.linalg.norm(bias) + 1e-8)

    def set_context(self, context: np.ndarray) -> None:
        """Set current context for relevance computation."""
        if len(context) != self.params.n_features:
            context = np.resize(context, self.params.n_features)
        self._context = context

    def set_content_priority(self, content_type: ContentType, priority: float) -> None:
        """Set priority weight for a content type."""
        self._content_priorities[content_type] = priority

    def reset(self) -> None:
        """Reset competition state."""
        self._attention_bias = np.zeros(self.params.n_features)
        self._context = np.zeros(self.params.n_features)
