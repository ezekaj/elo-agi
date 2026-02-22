"""
Episodic Memory: Personal experiences with rich context

Based on research showing hippocampus-dependent storage of autobiographical
events with spatial, temporal, and emotional context.

Brain region: Hippocampus
"""

import time
from typing import Optional, Any, List, Dict
from dataclasses import dataclass, field
import hashlib


@dataclass
class Episode:
    """Single episodic memory - a personal experience"""

    content: Any
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    vividness: float = 1.0  # Detail level 0-1
    retrieval_count: int = 0
    last_retrieval: Optional[float] = None
    episode_id: str = field(default="")
    strength: float = 1.0  # Consolidation strength

    def __post_init__(self):
        if not self.episode_id:
            content_str = str(self.content) + str(self.timestamp)
            self.episode_id = hashlib.md5(content_str.encode()).hexdigest()[:12]

    def __hash__(self):
        return hash(self.episode_id)

    def __eq__(self, other):
        if isinstance(other, Episode):
            return self.episode_id == other.episode_id
        return False


class EpisodicMemory:
    """
    Hippocampus - personal experiences, unlimited capacity

    Implements episodic memory with multiple retrieval routes:
    - Cue-based (pattern completion)
    - Context-dependent
    - Temporal
    - Emotion-congruent
    """

    def __init__(self):
        self._episodes: List[Episode] = []
        self._time_fn = time.time
        # Indices for fast lookup
        self._by_id: Dict[str, Episode] = {}
        self._by_context: Dict[str, List[Episode]] = {}

    def encode(
        self,
        experience: Any,
        context: Optional[Dict[str, Any]] = None,
        emotional_valence: float = 0.0,
        vividness: float = 1.0,
    ) -> Episode:
        """
        Create new episodic memory.

        Args:
            experience: What happened
            context: Where, when, who, etc.
            emotional_valence: Emotional coloring (-1 to +1)
            vividness: Level of detail (0-1)

        Returns:
            The created Episode
        """
        episode = Episode(
            content=experience,
            context=context or {},
            timestamp=self._time_fn(),
            emotional_valence=emotional_valence,
            vividness=vividness,
        )

        self._episodes.append(episode)
        self._by_id[episode.episode_id] = episode

        # Index by context keys
        for key, value in episode.context.items():
            context_key = f"{key}:{value}"
            if context_key not in self._by_context:
                self._by_context[context_key] = []
            self._by_context[context_key].append(episode)

        return episode

    def retrieve_by_cue(self, cue: Any, threshold: float = 0.3) -> List[Episode]:
        """
        Pattern completion from partial information.

        Args:
            cue: Partial information to match
            threshold: Minimum match strength (0-1)

        Returns:
            List of matching episodes sorted by match strength
        """
        matches = []

        for episode in self._episodes:
            match_strength = self._compute_match(episode, cue)
            if match_strength >= threshold:
                episode.retrieval_count += 1
                episode.last_retrieval = self._time_fn()
                matches.append((episode, match_strength))

        # Sort by match strength
        matches.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in matches]

    def _compute_match(self, episode: Episode, cue: Any) -> float:
        """Compute match strength between episode and cue"""
        if cue is None:
            return 0.0

        # Direct content match
        if self._content_matches(episode.content, cue):
            return 1.0

        # Partial string match
        if isinstance(cue, str):
            content_str = str(episode.content).lower()
            if cue.lower() in content_str:
                return 0.7

            # Check context values
            for value in episode.context.values():
                if cue.lower() in str(value).lower():
                    return 0.5

        # Dict cue - check context overlap
        if isinstance(cue, dict):
            overlap = sum(
                1 for k, v in cue.items() if k in episode.context and episode.context[k] == v
            )
            if overlap > 0:
                return overlap / len(cue)

        return 0.0

    def _content_matches(self, content: Any, cue: Any) -> bool:
        """Check if content matches cue"""
        try:
            import numpy as np

            if isinstance(content, np.ndarray) and isinstance(cue, np.ndarray):
                return np.array_equal(content, cue)
        except ImportError:
            pass
        return content == cue

    def retrieve_by_context(self, context: Dict[str, Any]) -> List[Episode]:
        """
        Context-dependent recall.

        Args:
            context: Context attributes to match

        Returns:
            Episodes matching the context
        """
        candidates = set()

        for key, value in context.items():
            context_key = f"{key}:{value}"
            if context_key in self._by_context:
                if not candidates:
                    candidates = set(self._by_context[context_key])
                else:
                    candidates &= set(self._by_context[context_key])

        for episode in candidates:
            episode.retrieval_count += 1
            episode.last_retrieval = self._time_fn()

        return list(candidates)

    def retrieve_by_time(
        self, start_time: Optional[float] = None, end_time: Optional[float] = None
    ) -> List[Episode]:
        """
        Temporal search.

        Args:
            start_time: Beginning of time range (None for no lower bound)
            end_time: End of time range (None for no upper bound)

        Returns:
            Episodes within time range, sorted by timestamp
        """
        matches = []

        for episode in self._episodes:
            if start_time is not None and episode.timestamp < start_time:
                continue
            if end_time is not None and episode.timestamp > end_time:
                continue
            matches.append(episode)
            episode.retrieval_count += 1
            episode.last_retrieval = self._time_fn()

        matches.sort(key=lambda e: e.timestamp)
        return matches

    def retrieve_by_emotion(
        self, valence_min: float = -1.0, valence_max: float = 1.0
    ) -> List[Episode]:
        """
        Mood-congruent recall.

        Args:
            valence_min: Minimum emotional valence
            valence_max: Maximum emotional valence

        Returns:
            Episodes within emotional range
        """
        matches = [
            ep for ep in self._episodes if valence_min <= ep.emotional_valence <= valence_max
        ]

        for episode in matches:
            episode.retrieval_count += 1
            episode.last_retrieval = self._time_fn()

        return matches

    def replay(self, episodes: Optional[List[Episode]] = None) -> None:
        """
        Consolidation through replay (during sleep).

        Strengthens memory traces.

        Args:
            episodes: Specific episodes to replay (None for all recent)
        """
        if episodes is None:
            # Replay recent episodes
            current_time = self._time_fn()
            day_seconds = 24 * 60 * 60
            episodes = [ep for ep in self._episodes if current_time - ep.timestamp < day_seconds]

        for episode in episodes:
            # Strengthen through replay
            episode.strength = min(1.0, episode.strength + 0.1)
            # Emotional memories strengthened more
            if abs(episode.emotional_valence) > 0.5:
                episode.strength = min(1.0, episode.strength + 0.05)

    def forget(self, episode: Episode, method: str = "decay") -> bool:
        """
        Remove or weaken memory.

        Args:
            episode: Episode to forget
            method: "decay" (weaken), "interference" (replace), "suppress" (block)

        Returns:
            True if memory affected
        """
        if episode not in self._episodes:
            return False

        if method == "decay":
            episode.strength *= 0.5
            episode.vividness *= 0.7
            if episode.strength < 0.1:
                return self._remove_episode(episode)
            return True

        elif method == "suppress":
            episode.strength *= 0.3
            return True

        elif method == "interference":
            return self._remove_episode(episode)

        return False

    def _remove_episode(self, episode: Episode) -> bool:
        """Remove episode from all indices"""
        if episode in self._episodes:
            self._episodes.remove(episode)
            del self._by_id[episode.episode_id]

            for key, episodes in self._by_context.items():
                if episode in episodes:
                    episodes.remove(episode)

            return True
        return False

    def update(self, episode: Episode, modification: Dict[str, Any]) -> Episode:
        """
        Reconsolidation - update reactivated memory.

        Args:
            episode: Episode to update
            modification: Changes to apply

        Returns:
            Updated episode
        """
        # Mark as recently retrieved (makes it labile)
        episode.last_retrieval = self._time_fn()

        # Apply modifications
        if "content" in modification:
            episode.content = modification["content"]
        if "context" in modification:
            episode.context.update(modification["context"])
        if "emotional_valence" in modification:
            episode.emotional_valence = modification["emotional_valence"]
        if "vividness" in modification:
            episode.vividness = modification["vividness"]

        return episode

    def get_recent(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes"""
        sorted_eps = sorted(self._episodes, key=lambda e: e.timestamp, reverse=True)
        return sorted_eps[:n]

    def get_strongest(self, n: int = 10) -> List[Episode]:
        """Get n strongest (most consolidated) episodes"""
        sorted_eps = sorted(self._episodes, key=lambda e: e.strength, reverse=True)
        return sorted_eps[:n]

    def __len__(self) -> int:
        return len(self._episodes)

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
