"""
Temporal Reasoning - Hippocampus Simulation

Sequence processing, time-based reasoning, and temporal pattern detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class TemporalRelation(Enum):
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    MEETS = "meets"
    STARTS = "starts"
    FINISHES = "finishes"
    EQUALS = "equals"


@dataclass
class TemporalEvent:
    """An event with temporal properties"""
    event_id: str
    start_time: float
    end_time: float
    content: Any = None
    properties: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps_with(self, other: 'TemporalEvent') -> bool:
        return self.start_time < other.end_time and other.start_time < self.end_time


@dataclass
class SequencePattern:
    """A detected pattern in a sequence"""
    pattern_type: str
    elements: List[Any]
    confidence: float
    period: Optional[float] = None
    parameters: Dict = field(default_factory=dict)


class SequenceMemory:
    """Store and retrieve temporal sequences"""

    def __init__(self, max_sequences: int = 1000):
        self.sequences: Dict[str, List[Any]] = {}
        self.sequence_metadata: Dict[str, Dict] = {}
        self.max_sequences = max_sequences

    def encode_sequence(self,
                        sequence_id: str,
                        items: List[Any],
                        metadata: Dict = None):
        """Store a sequence with its temporal order"""
        if len(self.sequences) >= self.max_sequences:
            oldest = min(self.sequence_metadata.items(),
                        key=lambda x: x[1].get('timestamp', 0))
            del self.sequences[oldest[0]]
            del self.sequence_metadata[oldest[0]]

        self.sequences[sequence_id] = list(items)
        self.sequence_metadata[sequence_id] = metadata or {}
        self.sequence_metadata[sequence_id]['timestamp'] = len(self.sequences)

    def retrieve_sequence(self, sequence_id: str) -> Optional[List[Any]]:
        """Retrieve a sequence by ID"""
        return self.sequences.get(sequence_id)

    def retrieve_by_cue(self, cue: Any, threshold: float = 0.5) -> List[Tuple[str, List[Any]]]:
        """Find sequences containing the cue"""
        matches = []
        for seq_id, sequence in self.sequences.items():
            if cue in sequence:
                matches.append((seq_id, sequence))
        return matches

    def compare_sequences(self,
                          seq1: List[Any],
                          seq2: List[Any]
                          ) -> float:
        """Compute similarity between two sequences using edit distance"""
        if not seq1 or not seq2:
            return 0.0

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        edit_distance = dp[m][n]
        max_len = max(m, n)
        similarity = 1.0 - (edit_distance / max_len)
        return similarity

    def find_pattern(self, sequence: List[Any]) -> List[SequencePattern]:
        """Detect patterns in a sequence"""
        patterns = []

        for period in range(1, len(sequence) // 2 + 1):
            is_repeating = True
            for i in range(period, len(sequence)):
                if sequence[i] != sequence[i % period]:
                    is_repeating = False
                    break

            if is_repeating:
                patterns.append(SequencePattern(
                    pattern_type='repeating',
                    elements=sequence[:period],
                    confidence=1.0,
                    period=period
                ))
                break

        if len(sequence) >= 3:
            diffs = [sequence[i+1] - sequence[i]
                    for i in range(len(sequence)-1)
                    if isinstance(sequence[i], (int, float))]

            if diffs and all(abs(d - diffs[0]) < 0.001 for d in diffs):
                patterns.append(SequencePattern(
                    pattern_type='arithmetic',
                    elements=sequence,
                    confidence=1.0,
                    parameters={'difference': diffs[0]}
                ))

            if diffs and len(diffs) >= 2:
                ratios = [sequence[i+1] / sequence[i]
                         for i in range(len(sequence)-1)
                         if isinstance(sequence[i], (int, float)) and sequence[i] != 0]
                if ratios and all(abs(r - ratios[0]) < 0.001 for r in ratios):
                    patterns.append(SequencePattern(
                        pattern_type='geometric',
                        elements=sequence,
                        confidence=1.0,
                        parameters={'ratio': ratios[0]}
                    ))

        return patterns


class TemporalReasoner:
    """
    Sequence and time-based reasoning.
    Simulates hippocampus temporal processing.
    """

    def __init__(self):
        self.events: Dict[str, TemporalEvent] = {}
        self.sequence_memory = SequenceMemory()
        self.current_time = 0.0

    def add_event(self, event: TemporalEvent):
        """Add an event to temporal memory"""
        self.events[event.event_id] = event

    def order_events(self, event_ids: List[str]) -> List[TemporalEvent]:
        """Establish temporal sequence of events"""
        events = [self.events[eid] for eid in event_ids if eid in self.events]
        return sorted(events, key=lambda e: e.start_time)

    def get_relation(self,
                     event1_id: str,
                     event2_id: str
                     ) -> Optional[TemporalRelation]:
        """Determine temporal relation between two events (Allen's interval algebra)"""
        if event1_id not in self.events or event2_id not in self.events:
            return None

        e1 = self.events[event1_id]
        e2 = self.events[event2_id]

        if e1.end_time < e2.start_time:
            return TemporalRelation.BEFORE
        if e1.start_time > e2.end_time:
            return TemporalRelation.AFTER
        if abs(e1.end_time - e2.start_time) < 0.001:
            return TemporalRelation.MEETS
        if (abs(e1.start_time - e2.start_time) < 0.001 and
            abs(e1.end_time - e2.end_time) < 0.001):
            return TemporalRelation.EQUALS
        if e1.start_time >= e2.start_time and e1.end_time <= e2.end_time:
            return TemporalRelation.DURING
        if abs(e1.start_time - e2.start_time) < 0.001 and e1.end_time < e2.end_time:
            return TemporalRelation.STARTS
        if abs(e1.end_time - e2.end_time) < 0.001 and e1.start_time > e2.start_time:
            return TemporalRelation.FINISHES
        if e1.overlaps_with(e2):
            return TemporalRelation.OVERLAPS

        return None

    def duration_estimation(self, event_id: str) -> Optional[float]:
        """Estimate duration of an event"""
        if event_id not in self.events:
            return None
        return self.events[event_id].duration

    def interval_reasoning(self,
                           event1_id: str,
                           event2_id: str
                           ) -> Optional[float]:
        """Calculate time between two events"""
        if event1_id not in self.events or event2_id not in self.events:
            return None

        e1 = self.events[event1_id]
        e2 = self.events[event2_id]

        if e1.end_time <= e2.start_time:
            return e2.start_time - e1.end_time
        elif e2.end_time <= e1.start_time:
            return -(e1.start_time - e2.end_time)
        else:
            return 0.0

    def predict_next(self,
                     sequence: List[Any],
                     n_predictions: int = 1
                     ) -> List[Any]:
        """Predict next elements in a sequence"""
        patterns = self.sequence_memory.find_pattern(sequence)

        if not patterns:
            if sequence:
                return [sequence[-1]] * n_predictions
            return []

        predictions = []
        pattern = patterns[0]

        if pattern.pattern_type == 'repeating':
            period = len(pattern.elements)
            for i in range(n_predictions):
                next_idx = (len(sequence) + i) % period
                predictions.append(pattern.elements[next_idx])

        elif pattern.pattern_type == 'arithmetic':
            diff = pattern.parameters['difference']
            last_val = sequence[-1]
            for i in range(n_predictions):
                predictions.append(last_val + diff * (i + 1))

        elif pattern.pattern_type == 'geometric':
            ratio = pattern.parameters['ratio']
            last_val = sequence[-1]
            for i in range(n_predictions):
                predictions.append(last_val * (ratio ** (i + 1)))

        return predictions

    def find_events_in_range(self,
                             start_time: float,
                             end_time: float
                             ) -> List[TemporalEvent]:
        """Find all events within a time range"""
        return [e for e in self.events.values()
                if e.start_time < end_time and e.end_time > start_time]

    def construct_timeline(self,
                           event_ids: List[str] = None
                           ) -> List[Tuple[float, str, str]]:
        """Build a timeline of events with start/end markers"""
        if event_ids is None:
            events = list(self.events.values())
        else:
            events = [self.events[eid] for eid in event_ids if eid in self.events]

        timeline = []
        for e in events:
            timeline.append((e.start_time, 'start', e.event_id))
            timeline.append((e.end_time, 'end', e.event_id))

        timeline.sort(key=lambda x: (x[0], x[1] == 'start'))
        return timeline

    def extrapolate_duration(self,
                             similar_events: List[str],
                             new_event_properties: Dict
                             ) -> float:
        """Estimate duration of a new event based on similar past events"""
        if not similar_events:
            return 1.0

        durations = [self.events[eid].duration
                    for eid in similar_events
                    if eid in self.events]

        if not durations:
            return 1.0

        return np.mean(durations)

    def detect_rhythms(self,
                       event_type: str = None,
                       min_occurrences: int = 3
                       ) -> List[Dict]:
        """Detect repeating temporal patterns"""
        if event_type:
            events = [e for e in self.events.values()
                     if e.properties.get('type') == event_type]
        else:
            events = list(self.events.values())

        if len(events) < min_occurrences:
            return []

        events = sorted(events, key=lambda e: e.start_time)
        intervals = [events[i+1].start_time - events[i].start_time
                    for i in range(len(events) - 1)]

        if not intervals:
            return []

        rhythms = []

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if std_interval < mean_interval * 0.2:
            rhythms.append({
                'type': 'regular',
                'period': mean_interval,
                'confidence': 1.0 - (std_interval / mean_interval)
            })

        return rhythms
