"""
Sleep Cycle Orchestrator

Coordinates all sleep components for complete sleep-wake processing:
- Sleep stage management (~90 min cycles)
- Memory replay during sleep
- Systems consolidation (hippocampal-cortical transfer)
- Synaptic homeostasis (downscaling)
- Dream generation during REM
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from .sleep_stages import SleepStage, SleepStageController
from .memory_replay import MemoryTrace, HippocampalReplay, ReplayScheduler
from .systems_consolidation import HippocampalCorticalDialogue
from .synaptic_homeostasis import SynapticHomeostasis, SelectiveConsolidation
from .dream_generator import DreamGenerator, DreamReport


@dataclass
class CycleStatistics:
    """Statistics for one sleep cycle"""

    cycle_number: int
    duration_minutes: float
    sws_minutes: float
    rem_minutes: float
    memories_replayed: int
    memories_consolidated: int
    memories_transferred: int
    dreams_generated: int
    synaptic_reduction: float


@dataclass
class NightStatistics:
    """Statistics for a full night of sleep"""

    total_cycles: int
    total_duration_minutes: float
    total_sws_minutes: float
    total_rem_minutes: float
    total_memories_replayed: int
    total_memories_consolidated: int
    total_memories_transferred: int
    total_dreams: int
    wake_synaptic_strength: float
    post_sleep_synaptic_strength: float
    dreams: List[DreamReport] = field(default_factory=list)


class SleepArchitecture:
    """Plans sleep stage sequence across the night.

    Key features:
    - SWS dominates early in the night
    - REM increases as night progresses
    - Cycle duration ~90 minutes
    """

    def __init__(self, target_sleep_hours: float = 8.0, cycle_duration_minutes: float = 90.0):
        self.target_sleep_hours = target_sleep_hours
        self.cycle_duration = cycle_duration_minutes

        # Calculate number of cycles
        self.n_cycles = int(target_sleep_hours * 60 / cycle_duration_minutes)

    def plan_night(self) -> List[Dict]:
        """Plan stage sequence for the night.

        Returns:
            List of cycle plans with stage durations
        """
        cycles = []

        for cycle_num in range(self.n_cycles):
            # SWS proportion decreases across night
            sws_factor = 0.8**cycle_num  # Exponential decrease

            # REM proportion increases across night
            rem_factor = 1.0 + 0.3 * cycle_num  # Linear increase

            # Calculate stage durations (in minutes)
            base_sws = 20 * sws_factor
            base_rem = 10 * rem_factor
            base_nrem2 = 25

            # Normalize to cycle duration
            total = base_sws + base_rem + base_nrem2 + 10  # +10 for NREM1
            scale = self.cycle_duration / total

            cycle = {
                "cycle_number": cycle_num,
                "nrem1_duration": 5 * scale,
                "nrem2_duration": base_nrem2 * scale,
                "sws_duration": base_sws * scale,
                "rem_duration": base_rem * scale,
            }
            cycles.append(cycle)

        return cycles

    def adjust_for_sleep_debt(self, debt_hours: float) -> List[Dict]:
        """Adjust sleep architecture for sleep debt.

        Sleep debt increases SWS pressure.

        Args:
            debt_hours: Hours of sleep debt

        Returns:
            Adjusted cycle plans
        """
        cycles = self.plan_night()

        # Sleep debt increases SWS in early cycles
        debt_factor = 1.0 + 0.2 * min(debt_hours, 8)  # Cap effect

        for i, cycle in enumerate(cycles):
            if i < 2:  # First two cycles most affected
                cycle["sws_duration"] *= debt_factor
                cycle["nrem2_duration"] *= 0.8  # Compensate

        return cycles

    def get_stage_targets(self) -> Dict[SleepStage, float]:
        """Get target minutes for each stage across night."""
        cycles = self.plan_night()

        targets = {
            SleepStage.NREM1: 0.0,
            SleepStage.NREM2: 0.0,
            SleepStage.SWS: 0.0,
            SleepStage.REM: 0.0,
        }

        for cycle in cycles:
            targets[SleepStage.NREM1] += cycle["nrem1_duration"]
            targets[SleepStage.NREM2] += cycle["nrem2_duration"]
            targets[SleepStage.SWS] += cycle["sws_duration"]
            targets[SleepStage.REM] += cycle["rem_duration"]

        return targets


class SleepCycleOrchestrator:
    """Orchestrates complete sleep-wake cycles.

    Coordinates:
    - Stage transitions
    - Memory replay
    - Systems consolidation
    - Synaptic homeostasis
    - Dream generation
    """

    def __init__(self, n_neurons: int = 100, memory_dim: int = 20):
        # Core components
        self.stage_controller = SleepStageController()
        self.replay_system = HippocampalReplay()
        self.replay_scheduler = ReplayScheduler(self.replay_system)
        self.consolidation = HippocampalCorticalDialogue()
        self.homeostasis = SynapticHomeostasis(n_neurons=n_neurons)
        self.selective_consolidation = SelectiveConsolidation(self.homeostasis)
        self.dream_generator = DreamGenerator()

        # Sleep architecture
        self.architecture = SleepArchitecture()

        # State
        self.is_sleeping = False
        self.current_cycle = 0
        self.total_sleep_time = 0.0

        # Statistics
        self.cycle_stats: List[CycleStatistics] = []

    def wake_encoding(
        self, experiences: List[np.ndarray], emotional_saliences: Optional[List[float]] = None
    ) -> List[MemoryTrace]:
        """Encode experiences during wake period.

        Args:
            experiences: List of experience patterns
            emotional_saliences: Emotional significance of each

        Returns:
            Created memory traces
        """
        if emotional_saliences is None:
            emotional_saliences = [0.0] * len(experiences)

        memories = []
        for exp, emotion in zip(experiences, emotional_saliences):
            trace = self.replay_system.encode_experience(pattern=exp, emotional_salience=emotion)

            # Also encode in hippocampus for consolidation
            self.consolidation.hippocampus.encode(trace)

            memories.append(trace)

        # Wake also causes synaptic potentiation
        # (simplified - just increase random weights)
        for _ in range(len(experiences)):
            pre = np.random.random(self.homeostasis.n_neurons) > 0.7
            post = np.random.random(self.homeostasis.n_neurons) > 0.7
            self.homeostasis.hebbian_potentiation(
                pre.astype(float), post.astype(float), learning_rate=0.01
            )

        return memories

    def start_sleep(self) -> None:
        """Begin sleep period."""
        self.is_sleeping = True
        self.stage_controller.start_sleep()
        self.current_cycle = 0

        # Tag important synapses before sleep
        self.selective_consolidation.tag_by_weight(percentile=90)

    def process_stage(self, stage: SleepStage, duration_minutes: float) -> Dict:
        """Process a single sleep stage.

        Args:
            stage: The sleep stage
            duration_minutes: Duration in minutes

        Returns:
            Stage processing statistics
        """
        stats = {
            "stage": stage.value,
            "duration": duration_minutes,
            "replayed": 0,
            "consolidated": 0,
            "transferred": 0,
            "dreams": 0,
            "synaptic_change": 0.0,
        }

        if stage == SleepStage.SWS:
            # SWS: Maximum consolidation, replay with ripples
            replayed = self.replay_scheduler.run_sws_replay(
                duration_minutes, slow_oscillation_phase="up"
            )
            stats["replayed"] = len(replayed)

            # Systems consolidation
            self.consolidation.initiate_dialogue(slow_osc_phase="up", spindle=True, ripple=True)
            consolidation_result = self.consolidation.run_consolidation_cycle(duration_minutes * 60)
            stats["consolidated"] = consolidation_result["consolidated"]
            stats["transferred"] = consolidation_result["transferred"]

            # Synaptic downscaling
            initial_strength = self.homeostasis.measure_total_strength()
            self.homeostasis.sleep_downscaling_step(dt=duration_minutes)
            final_strength = self.homeostasis.measure_total_strength()
            stats["synaptic_change"] = final_strength - initial_strength

        elif stage == SleepStage.NREM2:
            # NREM2: Moderate replay with spindles
            replayed = self.replay_scheduler.run_nrem2_replay(duration_minutes)
            stats["replayed"] = len(replayed)

            # Some consolidation
            self.consolidation.initiate_dialogue(slow_osc_phase="down", spindle=True, ripple=False)
            consolidation_result = self.consolidation.run_consolidation_cycle(duration_minutes * 60)
            stats["consolidated"] = consolidation_result["consolidated"]

            # Mild downscaling
            self.homeostasis.sleep_downscaling_step(dt=duration_minutes * 0.5)

        elif stage == SleepStage.REM:
            # REM: Emotional processing, dreams
            replayed = self.replay_scheduler.run_rem_replay(duration_minutes)
            stats["replayed"] = len(replayed)

            # Generate dreams from replayed memories
            if replayed:
                self.dream_generator.generate_dream(replayed, duration=duration_minutes)
                stats["dreams"] = 1

            # Minimal synaptic change during REM
            # (but important for emotional memory)

        elif stage == SleepStage.NREM1:
            # NREM1: Transition, minimal processing
            pass

        # Advance time
        self.stage_controller.advance_time(duration_minutes)
        self.total_sleep_time += duration_minutes

        return stats

    def run_cycle(self, cycle_plan: Dict) -> CycleStatistics:
        """Run one complete sleep cycle (~90 minutes).

        Args:
            cycle_plan: Plan for this cycle from SleepArchitecture

        Returns:
            Statistics for the cycle
        """
        self.current_cycle = cycle_plan["cycle_number"]

        cycle_stats = CycleStatistics(
            cycle_number=self.current_cycle,
            duration_minutes=0,
            sws_minutes=0,
            rem_minutes=0,
            memories_replayed=0,
            memories_consolidated=0,
            memories_transferred=0,
            dreams_generated=0,
            synaptic_reduction=0,
        )

        initial_strength = self.homeostasis.measure_total_strength()

        # Process each stage in order
        stages = [
            (SleepStage.NREM1, cycle_plan["nrem1_duration"]),
            (SleepStage.NREM2, cycle_plan["nrem2_duration"]),
            (SleepStage.SWS, cycle_plan["sws_duration"]),
            (SleepStage.NREM2, cycle_plan["nrem2_duration"] * 0.5),  # Back through NREM2
            (SleepStage.REM, cycle_plan["rem_duration"]),
        ]

        for stage, duration in stages:
            self.stage_controller.transition(stage)
            stats = self.process_stage(stage, duration)

            cycle_stats.duration_minutes += duration
            cycle_stats.memories_replayed += stats["replayed"]
            cycle_stats.memories_consolidated += stats["consolidated"]
            cycle_stats.memories_transferred += stats["transferred"]
            cycle_stats.dreams_generated += stats["dreams"]

            if stage == SleepStage.SWS:
                cycle_stats.sws_minutes += duration
            elif stage == SleepStage.REM:
                cycle_stats.rem_minutes += duration

        final_strength = self.homeostasis.measure_total_strength()
        cycle_stats.synaptic_reduction = initial_strength - final_strength

        self.cycle_stats.append(cycle_stats)
        return cycle_stats

    def sleep_consolidation(
        self, sleep_hours: float = 8.0, sleep_debt: float = 0.0
    ) -> NightStatistics:
        """Run complete night of sleep.

        Args:
            sleep_hours: Target sleep duration
            sleep_debt: Hours of accumulated sleep debt

        Returns:
            Statistics for the night
        """
        self.start_sleep()

        # Record initial state
        wake_strength = self.homeostasis.measure_total_strength()

        # Plan the night
        self.architecture.target_sleep_hours = sleep_hours
        if sleep_debt > 0:
            cycle_plans = self.architecture.adjust_for_sleep_debt(sleep_debt)
        else:
            cycle_plans = self.architecture.plan_night()

        # Run each cycle
        for plan in cycle_plans:
            self.run_cycle(plan)

        # Wake up
        self.wake_up()

        # Record final state
        post_sleep_strength = self.homeostasis.measure_total_strength()

        # Compile night statistics
        night_stats = NightStatistics(
            total_cycles=len(cycle_plans),
            total_duration_minutes=self.total_sleep_time,
            total_sws_minutes=sum(c.sws_minutes for c in self.cycle_stats),
            total_rem_minutes=sum(c.rem_minutes for c in self.cycle_stats),
            total_memories_replayed=sum(c.memories_replayed for c in self.cycle_stats),
            total_memories_consolidated=sum(c.memories_consolidated for c in self.cycle_stats),
            total_memories_transferred=sum(c.memories_transferred for c in self.cycle_stats),
            total_dreams=sum(c.dreams_generated for c in self.cycle_stats),
            wake_synaptic_strength=wake_strength,
            post_sleep_synaptic_strength=post_sleep_strength,
            dreams=self.dream_generator.dream_history.copy(),
        )

        return night_stats

    def wake_up(self) -> None:
        """End sleep period."""
        self.is_sleeping = False
        self.stage_controller.wake_up()

        # Clear synaptic tags
        self.selective_consolidation.clear_tags()

    def simulate_sleep_deprivation(self, skip_stages: List[SleepStage]) -> NightStatistics:
        """Simulate partial sleep with specific stages skipped.

        Args:
            skip_stages: Stages to skip (e.g., [SleepStage.SWS])

        Returns:
            Statistics showing effects of deprivation
        """
        self.start_sleep()
        wake_strength = self.homeostasis.measure_total_strength()

        cycle_plans = self.architecture.plan_night()

        for plan in cycle_plans:
            # Modify plan to skip certain stages
            if SleepStage.SWS in skip_stages:
                plan["sws_duration"] = 0
            if SleepStage.REM in skip_stages:
                plan["rem_duration"] = 0

            self.run_cycle(plan)

        self.wake_up()
        post_sleep_strength = self.homeostasis.measure_total_strength()

        return NightStatistics(
            total_cycles=len(cycle_plans),
            total_duration_minutes=self.total_sleep_time,
            total_sws_minutes=sum(c.sws_minutes for c in self.cycle_stats),
            total_rem_minutes=sum(c.rem_minutes for c in self.cycle_stats),
            total_memories_replayed=sum(c.memories_replayed for c in self.cycle_stats),
            total_memories_consolidated=sum(c.memories_consolidated for c in self.cycle_stats),
            total_memories_transferred=sum(c.memories_transferred for c in self.cycle_stats),
            total_dreams=sum(c.dreams_generated for c in self.cycle_stats),
            wake_synaptic_strength=wake_strength,
            post_sleep_synaptic_strength=post_sleep_strength,
            dreams=self.dream_generator.dream_history.copy(),
        )

    def get_consolidation_statistics(self) -> Dict:
        """Get current consolidation statistics."""
        return self.consolidation.get_statistics()

    def get_homeostasis_statistics(self) -> Dict:
        """Get current homeostasis statistics."""
        return self.homeostasis.get_statistics()

    def get_dream_statistics(self) -> Dict:
        """Get dream statistics."""
        return self.dream_generator.get_dream_statistics()

    def reset(self) -> None:
        """Reset all components."""
        self.stage_controller.reset()
        self.replay_system.reset()
        self.consolidation.reset()
        self.homeostasis.reset_statistics()
        self.selective_consolidation.clear_tags()
        self.dream_generator.reset()
        self.is_sleeping = False
        self.current_cycle = 0
        self.total_sleep_time = 0.0
        self.cycle_stats = []
