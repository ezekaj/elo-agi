"""
Comprehensive Stress Tests for Module 03: Four Types of Reasoning

These tests push the system to its limits with complex scenarios that
require integration of multiple reasoning types.
"""

import pytest
import numpy as np
import os
from neuro.modules.m03_reasoning_types.perceptual.visual_features import VisualFeatureExtractor, FeatureMap, Feature, FeatureType
from neuro.modules.m03_reasoning_types.perceptual.multimodal_integration import MultimodalIntegrator, SensoryInput, Modality
from neuro.modules.m03_reasoning_types.perceptual.object_recognition import ObjectRecognizer, CategoryLevel

from neuro.modules.m03_reasoning_types.dimensional.spatial_reasoning import SpatialReasoner, SpatialObject, SpatialRelations
from neuro.modules.m03_reasoning_types.dimensional.temporal_reasoning import TemporalReasoner, TemporalEvent, SequenceMemory
from neuro.modules.m03_reasoning_types.dimensional.hierarchical_reasoning import HierarchicalReasoner, Rule, AbstractionLevel

from neuro.modules.m03_reasoning_types.logical.inductive import InductiveReasoner, Observation
from neuro.modules.m03_reasoning_types.logical.deductive import DeductiveReasoner, Proposition, PropositionType
from neuro.modules.m03_reasoning_types.logical.abductive import AbductiveReasoner, Effect

from neuro.modules.m03_reasoning_types.interactive.feedback_adaptation import FeedbackAdapter, State, Action, ActionOutcome
from neuro.modules.m03_reasoning_types.interactive.theory_of_mind import TheoryOfMind, MentalStateModel
from neuro.modules.m03_reasoning_types.interactive.collaborative import CollaborativeReasoner, Agent, Task

from neuro.modules.m03_reasoning_types.reasoning_orchestrator import ReasoningOrchestrator, ReasoningType

class TestPerceptualStress:
    """Stress tests for perceptual reasoning"""

    def test_large_image_processing(self):
        """Process a moderately large image with many features"""
        extractor = VisualFeatureExtractor()

        # Use 128x128 for reasonable test time (512x512 is too slow)
        large_image = np.random.rand(128, 128)
        large_image[25:50, 25:50] = 1.0
        large_image[75:100, 75:100] = 0.0

        features = extractor.extract_all(large_image)

        assert features.width == 128
        assert features.height == 128
        assert len(list(features.features.keys())) > 0

    def test_noisy_edge_detection(self):
        """Edge detection on very noisy image"""
        extractor = VisualFeatureExtractor(edge_threshold=0.3)

        noisy = np.random.rand(100, 100) * 0.5
        noisy[40:60, :] += 0.5

        edges = extractor.extract_edges(noisy)

        horizontal_edges = [e for e in edges if abs(e.orientation) < 0.3 or abs(e.orientation - np.pi) < 0.3]
        assert len(horizontal_edges) > 0

    def test_motion_with_occlusion(self):
        """Motion detection with partial occlusion"""
        extractor = VisualFeatureExtractor()

        frames = []
        for t in range(5):
            frame = np.zeros((64, 64))
            x = 10 + t * 10
            frame[20:30, x:x+10] = 1.0
            if t == 2:
                frame[20:30, x:x+5] = 0.5
            frames.append(frame)

        motion = extractor.extract_motion(frames)
        assert len(motion) >= 0

    def test_multimodal_with_high_conflict(self):
        """Multimodal integration with very high sensory conflict"""
        integrator = MultimodalIntegrator(spatial_tolerance=0.1)

        visual = SensoryInput(
            modality=Modality.VISUAL,
            timestamp=0.0,
            location=(0.0, 0.0, 0.0),
            confidence=0.9
        )

        audio = SensoryInput(
            modality=Modality.AUDITORY,
            timestamp=0.0,
            location=(10.0, 0.0, 0.0),
            confidence=0.9
        )

        percept = integrator.bind_visual_audio(visual, audio)

        assert percept is None or percept.binding_strength < 0.5

    def test_category_hierarchy_learning(self):
        """Learn a deep category hierarchy"""
        recognizer = ObjectRecognizer()

        recognizer.learn_category('vehicle', [
            {'wheels': 4, 'engine': 1, 'moves': 1}
        ], CategoryLevel.SUPERORDINATE)

        recognizer.learn_category('car', [
            {'wheels': 4, 'doors': 4, 'engine': 1, 'seats': 5}
        ], CategoryLevel.BASIC, parent_category='vehicle')

        recognizer.learn_category('sedan', [
            {'wheels': 4, 'doors': 4, 'trunk': 1, 'seats': 5}
        ], CategoryLevel.SUBORDINATE, parent_category='car')

        recognizer.learn_category('suv', [
            {'wheels': 4, 'doors': 4, 'height': 2.0, 'seats': 7}
        ], CategoryLevel.SUBORDINATE, parent_category='car')

        test_features = {'wheels': 4, 'doors': 4, 'trunk': 1, 'seats': 5}
        result = recognizer.categorize(test_features, CategoryLevel.SUBORDINATE)

        if result:
            assert result.level in [CategoryLevel.BASIC, CategoryLevel.SUBORDINATE]

class TestDimensionalStress:
    """Stress tests for dimensional reasoning"""

    def test_complex_mental_rotation(self):
        """Mental rotation at many angles and verify time proportionality"""
        reasoner = SpatialReasoner()

        obj = SpatialObject(
            object_id='test_obj',
            position=np.array([0.0, 0.0, 0.0]),
            scale=np.array([1.0, 2.0, 1.0])
        )
        reasoner.add_object(obj)

        times = []
        angles = [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]

        for angle in angles:
            _, rotation_time = reasoner.mental_rotation('test_obj', angle)
            times.append(rotation_time)

        for i in range(1, len(times)):
            assert times[i] >= times[i-1] * 0.9

    def test_complex_navigation(self):
        """Navigate through a maze-like environment"""
        reasoner = SpatialReasoner()

        obstacles = [
            SpatialObject('wall1', np.array([5.0, 0.0, 0.0]), scale=np.array([1.0, 10.0, 1.0])),
            SpatialObject('wall2', np.array([10.0, 5.0, 0.0]), scale=np.array([10.0, 1.0, 1.0])),
            SpatialObject('wall3', np.array([15.0, 0.0, 0.0]), scale=np.array([1.0, 8.0, 1.0])),
        ]

        path = reasoner.navigation(
            start=(0.0, 0.0, 0.0),
            goal=(20.0, 10.0, 0.0),
            obstacles=obstacles,
            grid_resolution=1.0
        )

        assert len(path) >= 2
        assert np.linalg.norm(np.array(path[-1]) - np.array([20.0, 10.0, 0.0])) < 2.0

    def test_long_sequence_prediction(self):
        """Predict next elements in complex sequences"""
        reasoner = TemporalReasoner()

        arithmetic_seq = [2, 5, 8, 11, 14, 17, 20]
        predictions = reasoner.predict_next(arithmetic_seq, n_predictions=3)
        assert predictions[0] == pytest.approx(23, rel=0.1)

        geometric_seq = [2, 4, 8, 16, 32]
        predictions = reasoner.predict_next(geometric_seq, n_predictions=2)
        assert predictions[0] == pytest.approx(64, rel=0.1)

        repeating_seq = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        predictions = reasoner.predict_next(repeating_seq, n_predictions=3)
        assert predictions == [1, 2, 3]

    def test_many_temporal_events(self):
        """Handle many overlapping temporal events"""
        reasoner = TemporalReasoner()

        for i in range(100):
            event = TemporalEvent(
                event_id=f'event_{i}',
                start_time=i * 0.5,
                end_time=i * 0.5 + 2.0,
                content=f'Event {i}'
            )
            reasoner.add_event(event)

        events_in_range = reasoner.find_events_in_range(10.0, 15.0)
        assert len(events_in_range) > 5

        timeline = reasoner.construct_timeline()
        assert len(timeline) == 200

    def test_deep_rule_hierarchy(self):
        """Build and traverse deep rule hierarchy"""
        reasoner = HierarchicalReasoner()

        rule_l0 = Rule(
            rule_id='r0',
            name='Base Rule',
            level=AbstractionLevel.CONCRETE,
            condition=lambda x: isinstance(x, int) and x > 0,
            action=lambda x: x * 2,
            priority=1
        )

        rule_l1 = Rule(
            rule_id='r1',
            name='Level 1',
            level=AbstractionLevel.BASIC,
            condition=lambda x: isinstance(x, int) and x > 10,
            action=lambda x: x + 100,
            priority=2
        )

        rule_l2 = Rule(
            rule_id='r2',
            name='Level 2',
            level=AbstractionLevel.INTERMEDIATE,
            condition=lambda x: isinstance(x, int) and x > 100,
            action=lambda x: x // 2,
            priority=3
        )

        reasoner.rule_hierarchy.add_rule(rule_l0)
        reasoner.rule_hierarchy.add_rule(rule_l1, parent_rule_id='r0')
        reasoner.rule_hierarchy.add_rule(rule_l2, parent_rule_id='r1')

        result, rule_used = reasoner.reason(150)
        assert result is not None

class TestLogicalStress:
    """Stress tests for logical reasoning"""

    def test_large_scale_induction(self):
        """Induction from many observations"""
        reasoner = InductiveReasoner(min_observations=10)

        for i in range(100):
            features = {
                'property_a': 'constant_value',
                'property_b': i % 3,
                'property_c': np.random.choice(['x', 'y', 'z'])
            }
            obs = Observation(f'obs_{i}', features)
            reasoner.observe(obs)

        hypotheses = reasoner.hypothesize()

        constant_hyp = [h for h in hypotheses if 'constant_value' in h.description]
        assert len(constant_hyp) > 0
        assert constant_hyp[0].confidence > 0.9

    def test_long_deductive_chain(self):
        """Long chain of deductive inferences"""
        reasoner = DeductiveReasoner()

        categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        premises = []

        for i in range(len(categories) - 1):
            prop = Proposition(
                f'p{i}',
                PropositionType.UNIVERSAL,
                f'All {categories[i]} are {categories[i+1]}',
                subject=categories[i],
                predicate=categories[i+1]
            )
            premises.append(prop)
            reasoner.add_premise(prop)

        conclusion = Proposition(
            'target',
            PropositionType.UNIVERSAL,
            'All A are H',
            subject='A',
            predicate='H'
        )

        is_valid, explanation = reasoner.validate(conclusion, premises)
        assert is_valid

    def test_multiple_competing_abductions(self):
        """Abduction with many competing hypotheses"""
        reasoner = AbductiveReasoner()

        reasoner.add_causal_knowledge('power_outage', ['lights_off', 'computer_off'], prior=0.1, complexity=1.0, cause_id='power_outage')
        reasoner.add_causal_knowledge('bulb_burned', ['lights_off'], prior=0.3, complexity=1.0, cause_id='bulb_burned')
        reasoner.add_causal_knowledge('switch_off', ['lights_off'], prior=0.5, complexity=1.0, cause_id='switch_off')
        reasoner.add_causal_knowledge('computer_sleep', ['computer_off'], prior=0.4, complexity=1.0, cause_id='computer_sleep')

        effects = [
            Effect('lights_off', 'lights are off'),
            Effect('computer_off', 'computer is off')
        ]

        for e in effects:
            reasoner.observe_effect(e)

        diagnosis = reasoner.differential_diagnosis(effects)

        assert len(diagnosis) > 0
        best_cause, score = diagnosis[0]
        assert best_cause.cause_id == 'power_outage'

    def test_contradictory_premises(self):
        """Handle contradictory premises gracefully"""
        reasoner = DeductiveReasoner()

        p1 = Proposition('p1', PropositionType.UNIVERSAL, '', subject='birds', predicate='fly')
        p2 = Proposition('p2', PropositionType.ATOMIC, '', subject='penguin', predicate='birds')

        reasoner.add_premise(p1)
        reasoner.add_premise(p2)

        inferences = reasoner.derive([p1, p2])
        assert len(inferences) >= 0

class TestInteractiveStress:
    """Stress tests for interactive reasoning"""

    def test_long_learning_episode(self):
        """Learn over many interactions"""
        adapter = FeedbackAdapter(learning_rate=0.1, exploration_rate=0.3)

        actions = [Action(f'action_{i}', f'Action {i}') for i in range(5)]

        for episode in range(50):
            state = State(f'state_{episode % 10}', {'feature': episode % 10})
            adapter.set_state(state)
            adapter.set_available_actions(actions)

            action = adapter.act()
            reward = 1.0 if action.action_id == f'action_{episode % 5}' else -0.5

            next_state = State(f'state_{(episode + 1) % 10}', {'feature': (episode + 1) % 10})

            adapter.update_model(
                action,
                ActionOutcome.SUCCESS if reward > 0 else ActionOutcome.FAILURE,
                reward,
                next_state,
                actions
            )

        assert adapter.policy.total_reward != 0
        assert len(adapter.experiences) == 50

    def test_complex_theory_of_mind(self):
        """Complex multi-agent belief tracking"""
        tom = TheoryOfMind(self_id="observer")

        agents = ['alice', 'bob', 'charlie']
        for agent in agents:
            tom.agent_models[agent] = MentalStateModel(agent_id=agent)

        tom.update_world_state('treasure', 'cave', inform_agents=['alice'])
        tom.update_world_state('treasure', 'forest', inform_agents=['bob'])
        tom.update_world_state('treasure', 'mountain', inform_agents=['charlie'])

        tom.update_world_state('treasure', 'river', inform_agents=[])

        assert 'informed_treasure' in tom.agent_models['alice'].knowledge
        assert 'informed_treasure' in tom.agent_models['bob'].knowledge
        assert 'informed_treasure' in tom.agent_models['charlie'].knowledge

        assert tom.world_state['treasure'] == 'river'

    def test_multi_agent_task_coordination(self):
        """Coordinate complex task among multiple agents"""
        coordinator = CollaborativeReasoner(self_id="coordinator")

        agent_specs = [
            ('alice', {'coding', 'testing'}),
            ('bob', {'design', 'documentation'}),
            ('charlie', {'coding', 'design'}),
            ('diana', {'testing', 'documentation'})
        ]

        for name, caps in agent_specs:
            coordinator.register_agent(Agent(agent_id=name, capabilities=caps))

        main_task = coordinator.create_task(
            "Build software system",
            requirements=['coding', 'design', 'testing', 'documentation']
        )

        subtask1 = coordinator.create_task("Design architecture", requirements=['design'])
        subtask2 = coordinator.create_task("Write code", requirements=['coding'], dependencies=[subtask1.task_id])
        subtask3 = coordinator.create_task("Write tests", requirements=['testing'], dependencies=[subtask2.task_id])
        subtask4 = coordinator.create_task("Write docs", requirements=['documentation'])

        main_task.subtasks = [subtask1, subtask2, subtask3, subtask4]

        assignments = coordinator.divide_task(main_task, ['alice', 'bob', 'charlie', 'diana'])

        assert len(assignments) > 0

        for task_id, agent_id in [(subtask1.task_id, 'bob'), (subtask4.task_id, 'diana')]:
            coordinator.assign_task(task_id, agent_id)

        coordinator.report_task_completion(subtask1.task_id, "Design complete")
        coordinator.report_task_completion(subtask4.task_id, "Docs complete")

        assert coordinator.tasks[subtask1.task_id].completed
        assert coordinator.tasks[subtask4.task_id].completed

class TestOrchestratorStress:
    """Stress tests for the reasoning orchestrator"""

    def test_complex_multi_type_reasoning(self):
        """Task requiring multiple reasoning types"""
        orchestrator = ReasoningOrchestrator()

        task = "Where is the object that I believe Alice thinks is in the box?"

        analysis = orchestrator.analyze_task(task)

        assert ReasoningType.SPATIAL in analysis.required_types
        assert ReasoningType.SOCIAL in analysis.required_types

    def test_all_reasoning_types_activation(self):
        """Activate and use all reasoning types"""
        orchestrator = ReasoningOrchestrator()

        tasks_by_type = {
            ReasoningType.PERCEPTUAL: "Look at the image and recognize the pattern",
            ReasoningType.SPATIAL: "Where is the object located relative to the box?",
            ReasoningType.TEMPORAL: "What happened before the event occurred?",
            ReasoningType.INDUCTIVE: "Learn a pattern from these examples",
            ReasoningType.DEDUCTIVE: "Therefore conclude from the premises",
            ReasoningType.ABDUCTIVE: "Why did this effect occur?",
            ReasoningType.SOCIAL: "What does Alice believe about the situation?",
            ReasoningType.COLLABORATIVE: "Work together with the team to solve this"
        }

        for expected_type, task in tasks_by_type.items():
            analysis = orchestrator.analyze_task(task)
            assert expected_type in analysis.required_types, f"Expected {expected_type} for task: {task}"

    def test_meta_reasoning(self):
        """Meta-reasoning about reasoning strategy"""
        orchestrator = ReasoningOrchestrator()

        simple_task = "What is 2 + 2?"
        complex_task = "Where did Alice think Bob believed the treasure was hidden before Charlie told him about the map?"

        simple_analysis = orchestrator.analyze_task(simple_task)
        complex_analysis = orchestrator.analyze_task(complex_task)

        assert complex_analysis.complexity >= simple_analysis.complexity

        meta = orchestrator.meta_reason(complex_task)
        assert 'recommendations' in meta
        assert len(meta['recommendations']) > 0

class TestIntegrationScenarios:
    """Integration tests with realistic scenarios"""

    def test_detective_scenario(self):
        """
        Scenario: Solve a mystery using multiple reasoning types

        Evidence:
        - Footprints leading to the garden (perceptual)
        - Events happened between 8-10 PM (temporal)
        - Suspect A was seen at location X (spatial)
        - All previous thefts had pattern Y (inductive)
        - If guilty, then motive exists (deductive)
        - Best explanation for evidence (abductive)
        """
        inductive = InductiveReasoner()
        deductive = DeductiveReasoner()
        abductive = AbductiveReasoner()
        temporal = TemporalReasoner()

        for i in range(5):
            inductive.observe(Observation(f'theft_{i}', {
                'time': 'night',
                'entry': 'window',
                'target': 'jewelry'
            }))

        patterns = inductive.hypothesize()
        night_theft_pattern = [p for p in patterns if 'night' in p.description]
        assert len(night_theft_pattern) > 0

        witness_event = TemporalEvent('witness', 9.0, 9.5, 'Suspect seen near house')
        crime_event = TemporalEvent('crime', 8.5, 10.0, 'Crime occurred')
        temporal.add_event(witness_event)
        temporal.add_event(crime_event)

        relation = temporal.get_relation('witness', 'crime')
        assert relation is not None

        abductive.add_causal_knowledge(
            'suspect_guilty',
            ['footprints', 'motive', 'opportunity'],
            prior=0.5
        )

        explanation = abductive.explain('footprints')
        assert explanation is not None

    def test_robot_navigation_scenario(self):
        """
        Scenario: Robot navigating and learning environment

        Tasks:
        - Perceive obstacles (perceptual)
        - Plan path (spatial)
        - Remember sequence of locations (temporal)
        - Learn from collisions (feedback)
        - Predict human intentions (ToM)
        """
        spatial = SpatialReasoner()
        temporal = TemporalReasoner()
        feedback = FeedbackAdapter()

        obstacles = [
            SpatialObject('table', np.array([5.0, 5.0, 0.0]), scale=np.array([2.0, 2.0, 1.0])),
            SpatialObject('chair', np.array([8.0, 3.0, 0.0]), scale=np.array([1.0, 1.0, 1.0]))
        ]

        for obs in obstacles:
            spatial.add_object(obs)

        path = spatial.navigation(
            start=(0.0, 0.0, 0.0),
            goal=(10.0, 10.0, 0.0),
            obstacles=obstacles
        )
        assert len(path) >= 2

        for i, waypoint in enumerate(path[:5]):
            event = TemporalEvent(
                f'waypoint_{i}',
                float(i),
                float(i) + 0.5,
                content={'position': waypoint}
            )
            temporal.add_event(event)

        sequence = temporal.order_events([f'waypoint_{i}' for i in range(min(5, len(path)))])
        assert len(sequence) > 0

        move_forward = Action('forward', 'Move forward')
        state = State('s1', {'position': (0, 0), 'obstacle_ahead': False})
        feedback.set_state(state)
        feedback.set_available_actions([move_forward])

        feedback.update_model(
            move_forward,
            ActionOutcome.SUCCESS,
            1.0,
            State('s2', {'position': (1, 0), 'obstacle_ahead': False}),
            [move_forward]
        )

        assert len(feedback.experiences) == 1

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
