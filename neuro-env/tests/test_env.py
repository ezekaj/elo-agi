"""
Tests for the Neuro Environment Interface.

Covers:
- Base environment functionality
- Gym adapter
- Text world
- Dialogue environment
- Curriculum
- Experience buffer
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path
from neuro.modules.env.base_env import (
    NeuroEnvironment,
    EnvironmentConfig,
    StepResult,
    SimplePatternEnv,
)
from neuro.modules.env.text_world import TextWorld, TextWorldConfig, Room, Item, ProcGenTextWorld
from neuro.modules.env.dialogue_env import DialogueEnvironment, DialogueConfig, DialoguePartner
from neuro.modules.env.curriculum import (
    DevelopmentalCurriculum,
    Stage,
    CurriculumConfig,
    AdaptiveCurriculum,
)
from neuro.modules.env.experience_buffer import (
    ExperienceBuffer,
    Experience,
    Episode,
    SequenceBuffer,
    ConsolidationBuffer,
)

# =============================================================================
# Tests: Base Environment
# =============================================================================


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig."""

    def test_default_config(self):
        config = EnvironmentConfig()
        assert config.observation_dim == 64
        assert config.action_dim == 32
        assert config.max_steps == 1000

    def test_custom_config(self):
        config = EnvironmentConfig(observation_dim=128, action_dim=64)
        assert config.observation_dim == 128
        assert config.action_dim == 64


class TestSimplePatternEnv:
    """Tests for SimplePatternEnv."""

    def test_creation(self):
        env = SimplePatternEnv()
        assert env.config is not None

    def test_reset(self):
        env = SimplePatternEnv()
        obs, info = env.reset()
        assert len(obs) == env.config.observation_dim
        assert "pattern_idx" in info

    def test_step(self):
        env = SimplePatternEnv()
        env.reset()
        action = np.random.randn(env.config.action_dim)
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert len(result.observation) == env.config.observation_dim

    def test_multiple_steps(self):
        env = SimplePatternEnv()
        env.reset()
        for _ in range(10):
            action = np.random.randn(env.config.action_dim)
            result = env.step(action)
            assert not result.done or env._step_count >= env.config.max_steps

    def test_render(self):
        env = SimplePatternEnv()
        env.reset()
        output = env.render()
        assert output is not None
        assert isinstance(output, str)

    def test_statistics(self):
        env = SimplePatternEnv()
        env.reset()
        env.step(np.zeros(env.config.action_dim))
        stats = env.get_statistics()
        assert stats["step_count"] == 1
        assert stats["episode_count"] == 1


# =============================================================================
# Tests: Text World
# =============================================================================


class TestTextWorld:
    """Tests for TextWorld."""

    def test_creation(self):
        env = TextWorld()
        assert env._rooms is not None
        assert len(env._rooms) > 0

    def test_reset(self):
        env = TextWorld()
        obs, info = env.reset()
        assert len(obs) == env.config.observation_dim
        assert "description" in info

    def test_movement(self):
        env = TextWorld()
        env.reset()
        # Move north
        action = np.zeros(env.config.action_dim)
        action[0] = 1.0  # "north" is first action
        result = env.step(action)
        assert env._current_room == "hallway"

    def test_item_pickup(self):
        env = TextWorld()
        env.reset()
        # Take torch (first item in start room)
        action = np.zeros(env.config.action_dim)
        action[4] = 1.0  # "take" is index 4
        action[8] = 1.0  # First item
        result = env.step(action)
        assert "torch" in [i.name for i in env._inventory] or "take" in result.info.get(
            "command", ""
        )

    def test_render(self):
        env = TextWorld()
        env.reset()
        output = env.render()
        assert output is not None
        assert "room" in output.lower() or "door" in output.lower()

    def test_goal_completion(self):
        env = TextWorld()
        env.reset()
        # Navigate to treasury and take treasure
        # This is a multi-step goal
        steps = 0
        done = False
        while not done and steps < 100:
            action = np.random.randn(env.config.action_dim)
            result = env.step(action)
            done = result.done
            steps += 1
        # Should not crash and should have some history
        assert env._step_count > 0


class TestProcGenTextWorld:
    """Tests for procedurally generated text world."""

    def test_creation(self):
        env = ProcGenTextWorld(n_rooms=5)
        assert len(env._rooms) == 5

    def test_different_seeds(self):
        env1 = ProcGenTextWorld(n_rooms=5)
        env2 = ProcGenTextWorld(n_rooms=5)
        # Different random generations
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=123)
        # Should be different
        assert not np.allclose(obs1, obs2)


# =============================================================================
# Tests: Dialogue Environment
# =============================================================================


class TestDialoguePartner:
    """Tests for DialoguePartner."""

    def test_creation(self):
        partner = DialoguePartner()
        assert partner.persona is not None

    def test_respond(self):
        partner = DialoguePartner()
        response = partner.respond("Hello!")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_different_responses(self):
        partner = DialoguePartner()
        r1 = partner.respond("Hello!")
        r2 = partner.respond("What is AI?")
        # Different inputs should work
        assert isinstance(r1, str) and isinstance(r2, str)

    def test_reset(self):
        partner = DialoguePartner()
        partner.respond("Hello!")
        partner.reset()
        assert len(partner._context) == 0


class TestDialogueEnvironment:
    """Tests for DialogueEnvironment."""

    def test_creation(self):
        env = DialogueEnvironment()
        assert env._partner is not None

    def test_reset(self):
        env = DialogueEnvironment()
        obs, info = env.reset()
        assert len(obs) == env.config.observation_dim
        assert "message" in info

    def test_step(self):
        env = DialogueEnvironment()
        env.reset()
        action = np.random.randn(env.config.action_dim)
        result = env.step(action)
        assert "agent_message" in result.info
        assert "partner_response" in result.info

    def test_conversation(self):
        env = DialogueEnvironment()
        env.reset()
        for _ in range(5):
            action = np.random.randn(env.config.action_dim)
            result = env.step(action)
        assert len(env._history) == 11  # 1 initial + 5 * (agent + partner)

    def test_render(self):
        env = DialogueEnvironment()
        env.reset()
        env.step(np.random.randn(env.config.action_dim))
        output = env.render()
        assert output is not None


# =============================================================================
# Tests: Curriculum
# =============================================================================


class TestStage:
    """Tests for curriculum Stage."""

    def test_creation(self):
        stage = Stage(
            name="test",
            description="Test stage",
            env_class=SimplePatternEnv,
        )
        assert stage.name == "test"
        assert stage.status.value == "pending"

    def test_completion_check(self):
        stage = Stage(
            name="test",
            description="Test stage",
            env_class=SimplePatternEnv,
            min_episodes=10,
            success_threshold=0.8,
        )
        assert not stage.is_complete()
        stage.metrics.total_episodes = 100
        stage.metrics.success_rate = 0.9
        assert stage.is_complete()


class TestDevelopmentalCurriculum:
    """Tests for DevelopmentalCurriculum."""

    def test_creation(self):
        curriculum = DevelopmentalCurriculum()
        assert len(curriculum._stages) > 0

    def test_reset(self):
        curriculum = DevelopmentalCurriculum()
        obs, info = curriculum.reset()
        assert len(obs) == curriculum.current_stage.env_config.observation_dim
        assert "stage" in info

    def test_step(self):
        curriculum = DevelopmentalCurriculum()
        curriculum.reset()
        action = np.random.randn(curriculum.current_env.config.action_dim)
        result = curriculum.step(action)
        assert isinstance(result, StepResult)

    def test_progress(self):
        curriculum = DevelopmentalCurriculum()
        curriculum.reset()
        progress = curriculum.get_progress()
        assert "current_stage" in progress
        assert "total_stages" in progress

    def test_statistics(self):
        curriculum = DevelopmentalCurriculum()
        curriculum.reset()
        curriculum.step(np.zeros(curriculum.current_env.config.action_dim))
        stats = curriculum.get_statistics()
        assert "progress" in stats
        assert "current_stage_metrics" in stats

    def test_add_stage(self):
        curriculum = DevelopmentalCurriculum()
        initial_count = len(curriculum._stages)
        new_stage = Stage(
            name="new_stage",
            description="New stage",
            env_class=SimplePatternEnv,
        )
        curriculum.add_stage(new_stage)
        assert len(curriculum._stages) == initial_count + 1


class TestAdaptiveCurriculum:
    """Tests for AdaptiveCurriculum."""

    def test_creation(self):
        curriculum = AdaptiveCurriculum()
        assert curriculum._difficulty_level == 1.0

    def test_difficulty_property(self):
        curriculum = AdaptiveCurriculum()
        assert curriculum.difficulty == 1.0


# =============================================================================
# Tests: Experience Buffer
# =============================================================================


class TestExperience:
    """Tests for Experience dataclass."""

    def test_creation(self):
        exp = Experience(
            observation=np.zeros(64),
            action=np.zeros(32),
            reward=1.0,
            next_observation=np.zeros(64),
            done=False,
        )
        assert exp.reward == 1.0
        assert not exp.done


class TestExperienceBuffer:
    """Tests for ExperienceBuffer."""

    def test_creation(self):
        buffer = ExperienceBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert len(buffer) == 0

    def test_add(self):
        buffer = ExperienceBuffer()
        buffer.add(
            observation=np.zeros(64),
            action=np.zeros(32),
            reward=1.0,
            next_observation=np.zeros(64),
            done=False,
        )
        assert len(buffer) == 1

    def test_sample(self):
        buffer = ExperienceBuffer()
        for i in range(100):
            buffer.add(
                observation=np.ones(64) * i,
                action=np.zeros(32),
                reward=float(i),
                next_observation=np.ones(64) * (i + 1),
                done=(i % 10 == 9),
            )
        experiences, indices, weights = buffer.sample(10)
        assert len(experiences) == 10
        assert len(indices) == 10
        assert len(weights) == 10

    def test_sample_uniform(self):
        buffer = ExperienceBuffer()
        for i in range(50):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=1.0,
                next_observation=np.zeros(64),
                done=False,
            )
        experiences = buffer.sample_uniform(10)
        assert len(experiences) == 10

    def test_priority_update(self):
        buffer = ExperienceBuffer()
        for i in range(50):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=1.0,
                next_observation=np.zeros(64),
                done=False,
            )
        experiences, indices, _ = buffer.sample(10)
        new_priorities = np.ones(10) * 2.0
        buffer.update_priorities(indices, new_priorities)
        assert buffer._max_priority >= 2.0

    def test_episode_tracking(self):
        buffer = ExperienceBuffer()
        for i in range(20):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=1.0,
                next_observation=np.zeros(64),
                done=(i % 5 == 4),
            )
        # Should have 4 complete episodes (0-4, 5-9, 10-14, 15-19)
        assert len(buffer._episodes) == 4

    def test_sample_recent(self):
        buffer = ExperienceBuffer()
        for i in range(50):
            buffer.add(
                observation=np.ones(64) * i,
                action=np.zeros(32),
                reward=float(i),
                next_observation=np.ones(64) * (i + 1),
                done=False,
            )
        recent = buffer.sample_recent(10)
        assert len(recent) == 10
        # Most recent should have highest reward
        assert recent[0].reward == 49.0

    def test_clear(self):
        buffer = ExperienceBuffer()
        for i in range(10):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=1.0,
                next_observation=np.zeros(64),
                done=False,
            )
        buffer.clear()
        assert len(buffer) == 0

    def test_statistics(self):
        buffer = ExperienceBuffer()
        for i in range(10):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=float(i),
                next_observation=np.zeros(64),
                done=False,
            )
        stats = buffer.get_statistics()
        assert stats["size"] == 10
        assert stats["mean_reward"] == 4.5


class TestSequenceBuffer:
    """Tests for SequenceBuffer."""

    def test_creation(self):
        buffer = SequenceBuffer(sequence_length=16)
        assert buffer.sequence_length == 16

    def test_sample_sequences(self):
        buffer = SequenceBuffer(sequence_length=8)
        for i in range(100):
            buffer.add(
                observation=np.ones(64) * i,
                action=np.zeros(32),
                reward=float(i),
                next_observation=np.ones(64) * (i + 1),
                done=False,  # No episode boundaries
            )
        sequences = buffer.sample_sequences(5)
        assert len(sequences) > 0
        if sequences:
            assert len(sequences[0]) == 8


class TestConsolidationBuffer:
    """Tests for ConsolidationBuffer."""

    def test_creation(self):
        buffer = ConsolidationBuffer()
        assert buffer.consolidation_threshold == 10000

    def test_consolidation(self):
        buffer = ConsolidationBuffer(consolidation_threshold=50)
        for i in range(100):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=float(i),
                next_observation=np.zeros(64),
                done=False,
            )
            # Update some priorities
            if i < len(buffer._priorities):
                buffer._priorities[i] = float(i)

        result = buffer.consolidate()
        assert "consolidated" in result
        assert "forgotten" in result

    def test_should_consolidate(self):
        buffer = ConsolidationBuffer(consolidation_threshold=10)
        assert not buffer.should_consolidate()
        for i in range(15):
            buffer.add(
                observation=np.zeros(64),
                action=np.zeros(32),
                reward=1.0,
                next_observation=np.zeros(64),
                done=False,
            )
        assert buffer.should_consolidate()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full environment system."""

    def test_curriculum_with_buffer(self):
        """Test curriculum learning with experience buffer."""
        curriculum = DevelopmentalCurriculum()
        buffer = ExperienceBuffer(capacity=1000)

        obs, _ = curriculum.reset()

        for _ in range(50):
            action = np.random.randn(curriculum.current_env.config.action_dim)
            result = curriculum.step(action)

            buffer.add(
                observation=obs,
                action=action,
                reward=result.reward,
                next_observation=result.observation,
                done=result.done,
            )

            obs = result.observation
            if result.done:
                obs, _ = curriculum.reset()

        assert len(buffer) == 50
        assert curriculum._total_steps == 50

    def test_text_world_learning_loop(self):
        """Test text world in a learning loop."""
        env = TextWorld()
        buffer = ExperienceBuffer()

        for episode in range(3):
            obs, _ = env.reset()
            steps = 0

            while steps < 20:
                action = np.random.randn(env.config.action_dim)
                result = env.step(action)

                # Mark last step as done to complete episode
                is_done = steps == 19
                buffer.add(
                    observation=obs,
                    action=action,
                    reward=result.reward,
                    next_observation=result.observation,
                    done=is_done,
                )

                obs = result.observation
                steps += 1

        assert len(buffer) == 60  # 3 episodes * 20 steps
        assert len(buffer._episodes) == 3  # 3 complete episodes

    def test_dialogue_with_sampling(self):
        """Test dialogue environment with experience sampling."""
        env = DialogueEnvironment()
        buffer = ExperienceBuffer()

        obs, _ = env.reset()

        for _ in range(20):
            action = np.random.randn(env.config.action_dim)
            result = env.step(action)

            buffer.add(
                observation=obs,
                action=action,
                reward=result.reward,
                next_observation=result.observation,
                done=result.done,
            )

            obs = result.observation
            if result.done:
                break

        # Sample and check
        if len(buffer) >= 5:
            experiences, _, weights = buffer.sample(5)
            assert len(experiences) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
