"""
Moral Reasoning: Dual-process theory of morality.

Based on research from: https://www.nature.com/articles/s41598-024-68024-3

Dual-Process Theory of Morality:
| System       | Brain Region | Moral Framework                        | Speed |
|--------------|--------------|----------------------------------------|-------|
| Emotional    | VMPFC        | Deontological ("It's wrong to push")   | Fast  |
| Deliberative | DLPFC        | Utilitarian ("Save 5 by sacrificing 1")| Slow  |

VMPFC lesion patients:
- Make MORE utilitarian decisions
- Show emotional blunting
- Maintain intact intellectual abilities
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any
from enum import Enum
import numpy as np


class MoralFramework(Enum):
    """Moral reasoning frameworks."""

    DEONTOLOGICAL = "deontological"
    UTILITARIAN = "utilitarian"
    MIXED = "mixed"


class HarmType(Enum):
    """Types of harm in moral scenarios."""

    PERSONAL = "personal"  # Direct physical harm (pushing someone)
    IMPERSONAL = "impersonal"  # Indirect harm (flipping a switch)
    OMISSION = "omission"  # Harm by not acting
    ACTION = "action"  # Harm by acting


@dataclass
class MoralScenario:
    """
    A moral dilemma scenario.

    Example: Trolley problem variants
    """

    name: str
    description: str
    action_description: str
    harm_type: HarmType
    lives_saved: int
    lives_lost: int
    personal_involvement: float  # 0-1, how personally involved in harm
    emotional_intensity: float  # 0-1, emotional weight of scenario
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoralDecision:
    """Result of moral reasoning."""

    action_taken: bool
    framework_used: MoralFramework
    confidence: float
    deontological_weight: float
    utilitarian_weight: float
    reasoning: str
    emotional_response: float
    deliberation_time: float


class DeontologicalSystem:
    """
    Rule-based morality - VMPFC driven.

    Characteristics:
    - "It's wrong to push someone" regardless of outcome
    - Fast, intuitive, emotional
    - Personal harm triggers strong response
    - Rights-based thinking
    """

    def __init__(self, sensitivity: float = 1.0):
        self.sensitivity = sensitivity

        # Core moral rules (simplified)
        self.moral_rules: Dict[str, float] = {
            "do_not_kill": 1.0,
            "do_not_harm": 0.9,
            "do_not_lie": 0.7,
            "do_not_steal": 0.8,
            "help_others": 0.5,
            "keep_promises": 0.6,
        }

        # Personal harm aversion (key to trolley problem responses)
        self.personal_harm_aversion = 0.9

    def evaluate(self, scenario: MoralScenario) -> Tuple[float, str]:
        """
        Evaluate action based on deontological principles.

        Returns: (permissibility score from -1 to 1, reasoning)
        Higher = more permissible
        """
        # Base wrongness from lives lost (scaled to stay in reasonable range)
        base_wrongness = -0.3 * scenario.lives_lost

        # Personal involvement increases wrongness (key distinction)
        # This is what makes push worse than switch
        personal_factor = scenario.personal_involvement * self.personal_harm_aversion
        personal_wrongness = base_wrongness - personal_factor * 0.4

        # Emotional intensity adds additional weight
        emotional_factor = scenario.emotional_intensity * self.sensitivity * 0.2
        total_wrongness = personal_wrongness - emotional_factor

        # Use tanh to smoothly map to [-1, 1] while preserving ordering
        permissibility = np.tanh(total_wrongness * 2)

        # Generate reasoning
        if scenario.harm_type == HarmType.PERSONAL:
            reasoning = "Using someone as a means to an end violates their dignity"
        else:
            reasoning = "Causing harm is wrong, even for good outcomes"

        return permissibility, reasoning

    def is_personal_harm(self, scenario: MoralScenario) -> bool:
        """Check if scenario involves personal/direct harm."""
        return scenario.harm_type == HarmType.PERSONAL or scenario.personal_involvement > 0.5

    def emotional_response(self, scenario: MoralScenario) -> float:
        """
        Generate emotional response to scenario.

        Personal harm scenarios generate stronger negative emotions.
        """
        base_response = -scenario.lives_lost * 0.3

        if self.is_personal_harm(scenario):
            # Strong aversive response to personal harm
            personal_aversion = -0.5 * scenario.personal_involvement
            return base_response + personal_aversion

        return base_response


class UtilitarianSystem:
    """
    Outcome-based morality - DLPFC driven.

    Characteristics:
    - "Save 5 by sacrificing 1" = net positive
    - Slow, deliberative, calculating
    - Impersonal cost-benefit analysis
    - Consequentialist thinking
    """

    def __init__(self, calculation_precision: float = 1.0):
        self.calculation_precision = calculation_precision

        # Value weights
        self.life_value = 1.0
        self.suffering_weight = 0.5

    def evaluate(self, scenario: MoralScenario) -> Tuple[float, str]:
        """
        Evaluate action based on utilitarian calculus.

        Returns: (utility score from -1 to 1, reasoning)
        Higher = more permissible (greater net utility)
        """
        # Simple utility calculation
        lives_saved_value = scenario.lives_saved * self.life_value
        lives_lost_value = scenario.lives_lost * self.life_value

        # Net utility
        net_utility = lives_saved_value - lives_lost_value

        # Normalize to -1, 1 range
        max_possible = max(scenario.lives_saved, scenario.lives_lost) * self.life_value
        if max_possible > 0:
            normalized = net_utility / max_possible
        else:
            normalized = 0.0

        # Generate reasoning
        reasoning = f"Net utility: {scenario.lives_saved} saved - {scenario.lives_lost} lost = {scenario.lives_saved - scenario.lives_lost}"

        return normalized, reasoning

    def cost_benefit_analysis(self, scenario: MoralScenario) -> Dict[str, float]:
        """Detailed cost-benefit breakdown."""
        return {
            "lives_saved": scenario.lives_saved,
            "lives_lost": scenario.lives_lost,
            "net_lives": scenario.lives_saved - scenario.lives_lost,
            "utility_score": (scenario.lives_saved - scenario.lives_lost) * self.life_value,
        }


class MoralDilemmaProcessor:
    """
    Handles moral conflicts between deontological and utilitarian systems.

    Key finding from research:
    - Personal harm → VMPFC activation → deontological response
    - Impersonal harm → DLPFC activation → utilitarian response
    - VMPFC damage → more utilitarian responses
    """

    def __init__(self, vmpfc_intact: bool = True):
        self.deontological = DeontologicalSystem()
        self.utilitarian = UtilitarianSystem()
        self.vmpfc_intact = vmpfc_intact

        # Default weights
        self.base_deont_weight = 0.5
        self.base_util_weight = 0.5

    def process_dilemma(self, scenario: MoralScenario) -> MoralDecision:
        """
        Process a moral dilemma through both systems.

        Returns integrated decision.
        """
        # Both systems evaluate
        deont_score, deont_reasoning = self.deontological.evaluate(scenario)
        util_score, util_reasoning = self.utilitarian.evaluate(scenario)

        # Determine weights based on scenario and VMPFC integrity
        deont_weight, util_weight = self._compute_weights(scenario)

        # If VMPFC damaged, reduce deontological weight
        if not self.vmpfc_intact:
            deont_weight *= 0.3  # Emotional input reduced

        # Normalize weights
        total_weight = deont_weight + util_weight
        deont_weight /= total_weight
        util_weight /= total_weight

        # Combined score
        combined_score = deont_score * deont_weight + util_score * util_weight

        # Decision: positive score = action is permissible
        action_taken = combined_score > 0

        # Determine dominant framework
        if deont_weight > util_weight + 0.1:
            framework = MoralFramework.DEONTOLOGICAL
        elif util_weight > deont_weight + 0.1:
            framework = MoralFramework.UTILITARIAN
        else:
            framework = MoralFramework.MIXED

        # Confidence based on agreement between systems
        # Normalize to [0, 1] range (max disagreement is 2 when scores are -1 and +1)
        agreement = max(0, 1 - abs(deont_score - util_score) / 2)
        confidence = agreement * 0.5 + 0.5 * abs(combined_score)

        # Emotional response
        emotional = self.deontological.emotional_response(scenario)
        if not self.vmpfc_intact:
            emotional *= 0.2  # Emotional blunting

        # Deliberation time (utilitarian takes longer)
        deliberation = 100 * util_weight + 12 * deont_weight  # ms

        # Generate combined reasoning
        if framework == MoralFramework.DEONTOLOGICAL:
            reasoning = f"Deontological: {deont_reasoning}"
        elif framework == MoralFramework.UTILITARIAN:
            reasoning = f"Utilitarian: {util_reasoning}"
        else:
            reasoning = f"Mixed: {deont_reasoning}; {util_reasoning}"

        return MoralDecision(
            action_taken=action_taken,
            framework_used=framework,
            confidence=confidence,
            deontological_weight=deont_weight,
            utilitarian_weight=util_weight,
            reasoning=reasoning,
            emotional_response=emotional,
            deliberation_time=deliberation,
        )

    def _compute_weights(self, scenario: MoralScenario) -> Tuple[float, float]:
        """
        Compute weights for deontological vs utilitarian systems.

        Personal harm → higher deontological weight
        Impersonal harm → higher utilitarian weight
        """
        if self.deontological.is_personal_harm(scenario):
            # Personal scenarios engage VMPFC more
            deont_weight = 0.8
            util_weight = 0.2
        else:
            # Impersonal scenarios engage DLPFC more
            deont_weight = 0.3
            util_weight = 0.7

        # Emotional intensity increases deontological weight
        deont_weight += scenario.emotional_intensity * 0.2

        return deont_weight, util_weight


class VMPFCLesionModel:
    """
    Simulates effects of VMPFC damage on moral reasoning.

    Based on lesion studies showing VMPFC patients:
    - Make MORE utilitarian decisions
    - Show emotional blunting
    - Maintain intact intellectual abilities
    - More willing to sacrifice one to save many
    """

    def __init__(self):
        self.emotional_blunting = 0.8  # Reduced emotional response
        self.utilitarian_bias = 0.9  # Stronger utilitarian tendency
        self.intellectual_intact = True  # Reasoning preserved

        # Create lesioned processor
        self.processor = MoralDilemmaProcessor(vmpfc_intact=False)
        self.processor.deontological.sensitivity = 1 - self.emotional_blunting

    def process_moral_dilemma(self, scenario: MoralScenario) -> MoralDecision:
        """
        Process dilemma with simulated VMPFC damage.
        """
        decision = self.processor.process_dilemma(scenario)

        # Override to show lesion effects
        decision.emotional_response *= 1 - self.emotional_blunting

        return decision

    def compare_with_healthy(self, scenario: MoralScenario) -> Dict[str, MoralDecision]:
        """
        Compare lesioned vs healthy processing.
        """
        healthy_processor = MoralDilemmaProcessor(vmpfc_intact=True)

        return {
            "healthy": healthy_processor.process_dilemma(scenario),
            "lesioned": self.process_moral_dilemma(scenario),
        }


# Pre-built scenarios for testing


def create_trolley_switch() -> MoralScenario:
    """
    Classic trolley problem: flip switch to divert trolley.

    Impersonal harm - typically triggers utilitarian response.
    """
    return MoralScenario(
        name="Trolley Switch",
        description="A trolley is heading toward 5 people. You can flip a switch to divert it to a side track where it will hit 1 person.",
        action_description="Flip the switch",
        harm_type=HarmType.IMPERSONAL,
        lives_saved=5,
        lives_lost=1,
        personal_involvement=0.1,
        emotional_intensity=0.4,
    )


def create_trolley_push() -> MoralScenario:
    """
    Footbridge variant: push someone to stop trolley.

    Personal harm - typically triggers deontological response.
    """
    return MoralScenario(
        name="Trolley Push",
        description="A trolley is heading toward 5 people. You can push a large man off a bridge to stop it. He will die but the 5 will be saved.",
        action_description="Push the man",
        harm_type=HarmType.PERSONAL,
        lives_saved=5,
        lives_lost=1,
        personal_involvement=0.9,
        emotional_intensity=0.9,
    )


def create_crying_baby() -> MoralScenario:
    """
    Crying baby dilemma: smother baby to save village.

    Very personal, highly emotional.
    """
    return MoralScenario(
        name="Crying Baby",
        description="Enemy soldiers are searching for your group. Your baby starts crying. You can smother the baby to save 20 people, or risk everyone dying.",
        action_description="Smother the baby",
        harm_type=HarmType.PERSONAL,
        lives_saved=20,
        lives_lost=1,
        personal_involvement=1.0,
        emotional_intensity=1.0,
    )
