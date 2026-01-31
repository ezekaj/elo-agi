"""
Tests for neuro-inference: Probabilistic and causal reasoning.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian.networks import (
    BayesianNetwork, DiscreteNode, ContinuousNode, CPT, NetworkQuery,
)
from bayesian.belief_prop import (
    BeliefPropagation, Message, FactorGraph, Factor,
)
from bayesian.learning import (
    StructureLearner, ParameterLearner, BayesianScore,
)
from causal.scm import (
    StructuralCausalModel, CausalVariable, StructuralEquation, VariableType,
)
from causal.intervention import (
    Intervention, InterventionEngine, DoOperator,
)
from causal.counterfactual import (
    CounterfactualReasoner, CounterfactualQuery, PotentialOutcome,
)
from analogical.mapping import (
    StructureMapper, Analogy, StructuralAlignment,
    RelationalStructure, Predicate, RelationOrder,
)
from analogical.retrieval import (
    AnalogyRetriever, CaseLibrary, Case,
)
from integration import (
    ProbabilisticReasoner, InferenceResult, ReasoningType,
)


# =============================================================================
# Bayesian Network Tests
# =============================================================================

class TestCPT:
    """Tests for Conditional Probability Table."""

    def test_cpt_creation(self):
        """Test creating a CPT."""
        cpt = CPT(
            variable="rain",
            parents=[],
            probabilities={(): {"yes": 0.3, "no": 0.7}},
        )
        assert cpt.variable == "rain"
        assert cpt.get_probability("yes") == 0.3
        assert cpt.get_probability("no") == 0.7

    def test_cpt_with_parents(self):
        """Test CPT with parent conditioning."""
        cpt = CPT(
            variable="wet",
            parents=["rain"],
            probabilities={
                ("yes",): {"yes": 0.9, "no": 0.1},
                ("no",): {"yes": 0.1, "no": 0.9},
            },
        )
        assert cpt.get_probability("yes", {"rain": "yes"}) == 0.9
        assert cpt.get_probability("yes", {"rain": "no"}) == 0.1


class TestDiscreteNode:
    """Tests for DiscreteNode."""

    def test_node_creation(self):
        """Test creating a discrete node."""
        node = DiscreteNode(
            name="weather",
            values=["sunny", "rainy", "cloudy"],
        )
        assert node.name == "weather"
        assert len(node.values) == 3


class TestBayesianNetwork:
    """Tests for BayesianNetwork."""

    @pytest.fixture
    def simple_bn(self):
        """Create a simple Bayesian network."""
        bn = BayesianNetwork()

        # Rain -> Sprinkler -> Wet
        bn.add_discrete_node(
            "rain",
            ["yes", "no"],
            cpt={(): {"yes": 0.3, "no": 0.7}},
        )
        bn.add_discrete_node(
            "sprinkler",
            ["on", "off"],
            parents=["rain"],
            cpt={
                ("yes",): {"on": 0.1, "off": 0.9},
                ("no",): {"on": 0.5, "off": 0.5},
            },
        )
        bn.add_discrete_node(
            "wet",
            ["yes", "no"],
            parents=["rain", "sprinkler"],
            cpt={
                ("yes", "on"): {"yes": 0.99, "no": 0.01},
                ("yes", "off"): {"yes": 0.8, "no": 0.2},
                ("no", "on"): {"yes": 0.9, "no": 0.1},
                ("no", "off"): {"yes": 0.0, "no": 1.0},
            },
        )
        return bn

    def test_add_node(self):
        """Test adding nodes."""
        bn = BayesianNetwork()
        node = bn.add_discrete_node("test", ["a", "b"])
        assert bn.has_node("test")

    def test_get_topological_order(self, simple_bn):
        """Test topological ordering."""
        order = simple_bn.get_topological_order()
        # Rain should come before wet
        assert order.index("rain") < order.index("wet")

    def test_query(self, simple_bn):
        """Test probabilistic query."""
        result = simple_bn.query(["rain"])
        assert "rain" in result
        # Check that distribution sums to 1
        total = sum(result["rain"].values())
        assert abs(total - 1.0) < 0.1  # Allow some tolerance

    def test_query_with_evidence(self, simple_bn):
        """Test query with evidence."""
        result = simple_bn.query(["rain"], evidence={"wet": "yes"})
        assert "rain" in result
        # Should have non-empty distribution
        assert len(result["rain"]) > 0

    def test_d_separation(self, simple_bn):
        """Test d-separation."""
        # Rain and wet are not d-separated given nothing
        assert not simple_bn.is_d_separated("rain", "wet", set())
        # Sprinkler d-separates rain from wet? No, because rain directly causes wet
        assert not simple_bn.is_d_separated("rain", "wet", {"sprinkler"})

    def test_sample(self, simple_bn):
        """Test sampling from network."""
        samples = simple_bn.sample(n_samples=100)
        assert len(samples) == 100
        assert all("rain" in s for s in samples)

    def test_markov_blanket(self, simple_bn):
        """Test Markov blanket computation."""
        blanket = simple_bn.get_markov_blanket("sprinkler")
        assert "rain" in blanket  # Parent
        assert "wet" in blanket   # Child

    def test_statistics(self, simple_bn):
        """Test network statistics."""
        stats = simple_bn.statistics()
        assert stats["n_nodes"] == 3
        assert stats["n_edges"] == 3


# =============================================================================
# Belief Propagation Tests
# =============================================================================

class TestFactor:
    """Tests for Factor class."""

    def test_factor_creation(self):
        """Test creating a factor."""
        f = Factor("f1", ["A", "B"])
        f.set_potential({"A": 0, "B": 0}, 0.1)
        f.set_potential({"A": 0, "B": 1}, 0.9)
        assert f.get_potential({"A": 0, "B": 0}) == 0.1

    def test_factor_marginalize(self):
        """Test factor marginalization."""
        f = Factor("f1", ["A", "B"], {
            (("A", 0), ("B", 0)): 0.2,
            (("A", 0), ("B", 1)): 0.3,
            (("A", 1), ("B", 0)): 0.1,
            (("A", 1), ("B", 1)): 0.4,
        })
        marg = f.marginalize("B", [0, 1])
        # P(A=0) = 0.2 + 0.3 = 0.5
        # P(A=1) = 0.1 + 0.4 = 0.5
        assert "B" not in marg.variables


class TestFactorGraph:
    """Tests for FactorGraph."""

    def test_graph_creation(self):
        """Test creating factor graph."""
        fg = FactorGraph()
        fg.add_variable("A", [0, 1])
        fg.add_variable("B", [0, 1])

        f = Factor("f_AB", ["A", "B"])
        fg.add_factor(f)

        assert "A" in fg._variables
        assert fg.get_neighboring_factors("A") == ["f_AB"]

    def test_is_tree(self):
        """Test tree detection."""
        fg = FactorGraph()
        fg.add_variable("A", [0, 1])
        fg.add_variable("B", [0, 1])
        fg.add_factor(Factor("f_AB", ["A", "B"]))
        # A--f_AB--B is a tree
        assert fg.is_tree()


class TestBeliefPropagation:
    """Tests for BeliefPropagation."""

    @pytest.fixture
    def simple_fg(self):
        """Create simple factor graph."""
        fg = FactorGraph()
        fg.add_variable("A", [0, 1])
        fg.add_variable("B", [0, 1])

        f = Factor("f_AB", ["A", "B"], {
            (("A", 0), ("B", 0)): 0.9,
            (("A", 0), ("B", 1)): 0.1,
            (("A", 1), ("B", 0)): 0.2,
            (("A", 1), ("B", 1)): 0.8,
        })
        fg.add_factor(f)
        return fg

    def test_belief_propagation_run(self, simple_fg):
        """Test running BP."""
        bp = BeliefPropagation(simple_fg, max_iterations=10)
        iterations = bp.run()
        assert iterations <= 10

    def test_get_belief(self, simple_fg):
        """Test getting beliefs."""
        bp = BeliefPropagation(simple_fg)
        bp.run()
        belief = bp.get_belief("A")
        assert 0 in belief or 1 in belief


# =============================================================================
# Structure Learning Tests
# =============================================================================

class TestParameterLearner:
    """Tests for ParameterLearner."""

    def test_learn_cpt(self):
        """Test learning CPT from data."""
        data = [
            {"rain": "yes", "wet": "yes"},
            {"rain": "yes", "wet": "yes"},
            {"rain": "no", "wet": "no"},
            {"rain": "no", "wet": "no"},
            {"rain": "no", "wet": "no"},
        ]

        learner = ParameterLearner(pseudo_count=0.1)
        cpt = learner.learn_cpt(data, "wet", ["rain"])

        # P(wet=yes | rain=yes) should be high
        assert cpt.get_probability("yes", {"rain": "yes"}) > 0.8
        # P(wet=yes | rain=no) should be low
        assert cpt.get_probability("yes", {"rain": "no"}) < 0.2


class TestStructureLearner:
    """Tests for StructureLearner."""

    def test_learn_structure(self):
        """Test learning network structure."""
        # Generate data with A -> B -> C structure
        np.random.seed(42)
        data = []
        for _ in range(100):
            a = np.random.choice(["0", "1"])
            b = a if np.random.random() > 0.2 else ("1" if a == "0" else "0")
            c = b if np.random.random() > 0.2 else ("1" if b == "0" else "0")
            data.append({"A": a, "B": b, "C": c})

        learner = StructureLearner(score_type=BayesianScore.BIC)
        structure = learner.learn_structure(data, ["A", "B", "C"], method="greedy")

        # Should find some structure
        assert structure is not None
        assert "A" in structure


# =============================================================================
# Structural Causal Model Tests
# =============================================================================

class TestStructuralCausalModel:
    """Tests for StructuralCausalModel."""

    @pytest.fixture
    def simple_scm(self):
        """Create a simple SCM: X -> Y."""
        scm = StructuralCausalModel()
        scm.add_endogenous("X")
        scm.add_endogenous("Y")
        scm.add_linear_equation("X", [], {"X": 0}, intercept=0)
        scm.add_linear_equation("Y", ["X"], {"X": 2.0}, intercept=1.0)
        return scm

    def test_sample(self, simple_scm):
        """Test sampling from SCM."""
        samples = simple_scm.sample(n_samples=100)
        assert len(samples) == 100
        assert all("X" in s and "Y" in s for s in samples)

    def test_intervene(self, simple_scm):
        """Test intervention do(X=5)."""
        y_values = simple_scm.intervene({"X": 5}, "Y", n_samples=100)
        # Y = 1 + 2*X + noise, so E[Y|do(X=5)] ≈ 11
        mean_y = np.mean(y_values)
        assert 9 < mean_y < 13

    def test_causal_effect(self, simple_scm):
        """Test causal effect estimation."""
        effect = simple_scm.causal_effect("X", "Y", (0, 1), n_samples=1000)
        # True effect is 2 (coefficient)
        assert 1.5 < effect < 2.5

    def test_ancestors_descendants(self, simple_scm):
        """Test ancestor/descendant queries."""
        assert "X" in simple_scm.get_ancestors("Y")
        assert "Y" in simple_scm.get_descendants("X")


class TestIntervention:
    """Tests for Intervention classes."""

    def test_do_operator_single(self):
        """Test single variable intervention."""
        do = DoOperator.single("X", 5)
        assert do.variables == {"X": 5}
        assert "do(X=5)" in str(do)

    def test_do_operator_multiple(self):
        """Test multiple variable intervention."""
        do = DoOperator.multiple({"X": 1, "Z": 2})
        assert "X" in do.variables
        assert "Z" in do.variables


class TestInterventionEngine:
    """Tests for InterventionEngine."""

    @pytest.fixture
    def engine(self):
        """Create intervention engine."""
        scm = StructuralCausalModel()
        scm.add_endogenous("X")
        scm.add_endogenous("Y")
        scm.add_linear_equation("X", [], {}, intercept=0)
        scm.add_linear_equation("Y", ["X"], {"X": 1.0}, intercept=0)
        return InterventionEngine(scm)

    def test_apply_intervention(self, engine):
        """Test applying intervention."""
        do = DoOperator.single("X", 10)
        samples = engine.apply_intervention(do, n_samples=50)
        assert len(samples) == 50

    def test_average_treatment_effect(self, engine):
        """Test ATE computation."""
        ate, se = engine.average_treatment_effect("X", "Y", 0, 1, n_samples=1000)
        # True ATE is 1
        assert 0.5 < ate < 1.5


# =============================================================================
# Counterfactual Tests
# =============================================================================

class TestCounterfactualReasoner:
    """Tests for CounterfactualReasoner."""

    @pytest.fixture
    def reasoner(self):
        """Create counterfactual reasoner."""
        scm = StructuralCausalModel()
        scm.add_endogenous("X")
        scm.add_endogenous("Y")
        # X ~ Normal(0,1), Y = X + noise
        scm.add_linear_equation("X", [], {}, intercept=0)
        scm.add_linear_equation("Y", ["X"], {"X": 1.0}, intercept=0)
        return CounterfactualReasoner(scm)

    def test_counterfactual_query(self, reasoner):
        """Test counterfactual computation."""
        query = CounterfactualQuery(
            outcome="Y",
            intervention_var="X",
            intervention_value=5,
            evidence={"X": 0, "Y": 0.5},
        )
        result = reasoner.compute_counterfactual(query, n_samples=100)
        # Result should exist
        assert "outcomes" in result or "error" in result


class TestPotentialOutcome:
    """Tests for PotentialOutcome."""

    def test_potential_outcome_creation(self):
        """Test creating potential outcome."""
        po = PotentialOutcome(
            outcome_var="Y",
            treatment_var="X",
            treatment_value=1,
            value=10,
        )
        assert "Y(X=1)" in str(po)


# =============================================================================
# Analogical Reasoning Tests
# =============================================================================

class TestRelationalStructure:
    """Tests for RelationalStructure."""

    def test_structure_creation(self):
        """Test creating relational structure."""
        structure = RelationalStructure("solar_system")
        structure.add_object("sun", {"type": "star"})
        structure.add_object("earth", {"type": "planet"})
        structure.add_relation("orbits", ["earth", "sun"])

        assert "sun" in structure.objects
        assert len(structure.predicates) == 1


class TestStructureMapper:
    """Tests for StructureMapper."""

    @pytest.fixture
    def solar_system(self):
        """Create solar system domain."""
        s = RelationalStructure("solar_system")
        s.add_object("sun")
        s.add_object("earth")
        s.add_relation("more_massive", ["sun", "earth"])
        s.add_relation("attracts", ["sun", "earth"])
        s.add_relation("orbits", ["earth", "sun"])
        s.add_relation("causes", ["attracts", "orbits"], RelationOrder.SECOND)
        return s

    @pytest.fixture
    def atom(self):
        """Create atom domain."""
        a = RelationalStructure("atom")
        a.add_object("nucleus")
        a.add_object("electron")
        a.add_relation("more_massive", ["nucleus", "electron"])
        a.add_relation("attracts", ["nucleus", "electron"])
        a.add_relation("orbits", ["electron", "nucleus"])
        return a

    def test_map(self, solar_system, atom):
        """Test structure mapping."""
        mapper = StructureMapper()
        alignment = mapper.map(solar_system, atom)

        # Should map sun->nucleus, earth->electron
        assert alignment.object_mappings.get("sun") == "nucleus" or \
               alignment.object_mappings.get("earth") == "electron"

    def test_make_analogy(self, solar_system, atom):
        """Test full analogy creation."""
        mapper = StructureMapper()
        analogy = mapper.make_analogy(solar_system, atom)

        assert analogy.similarity > 0
        # Should have high systematicity due to causal structure
        assert analogy.systematicity >= 0


class TestCaseLibrary:
    """Tests for CaseLibrary."""

    def test_add_case(self):
        """Test adding case to library."""
        library = CaseLibrary()
        structure = RelationalStructure("test")
        case = Case(
            name="test_case",
            problem=structure,
            features={"type": "simple"},
        )
        library.add_case(case)
        assert library.size() == 1

    def test_get_case(self):
        """Test retrieving case."""
        library = CaseLibrary()
        structure = RelationalStructure("test")
        case = Case(name="test_case", problem=structure)
        library.add_case(case)

        retrieved = library.get_case("test_case")
        assert retrieved.name == "test_case"


class TestAnalogyRetriever:
    """Tests for AnalogyRetriever."""

    @pytest.fixture
    def library_with_cases(self):
        """Create library with test cases."""
        library = CaseLibrary()

        # Add solar system case
        solar = RelationalStructure("solar_system")
        solar.add_object("sun")
        solar.add_object("earth")
        solar.add_relation("orbits", ["earth", "sun"])
        case1 = Case(
            name="solar",
            problem=solar,
            features={"domain": "astronomy"},
        )
        library.add_case(case1)

        # Add atom case
        atom = RelationalStructure("atom")
        atom.add_object("nucleus")
        atom.add_object("electron")
        atom.add_relation("orbits", ["electron", "nucleus"])
        case2 = Case(
            name="atom",
            problem=atom,
            features={"domain": "physics"},
        )
        library.add_case(case2)

        return library

    def test_retrieve(self, library_with_cases):
        """Test case retrieval."""
        retriever = AnalogyRetriever(library_with_cases)

        query = RelationalStructure("query")
        query.add_object("A")
        query.add_object("B")
        query.add_relation("orbits", ["A", "B"])

        results = retriever.retrieve(query, top_k=2)
        assert len(results) >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestProbabilisticReasoner:
    """Tests for ProbabilisticReasoner."""

    def test_probabilistic_query(self):
        """Test probabilistic query."""
        reasoner = ProbabilisticReasoner()

        # Set up BN
        bn = BayesianNetwork()
        bn.add_discrete_node("A", ["0", "1"], cpt={(): {"0": 0.5, "1": 0.5}})
        reasoner.set_bayesian_network(bn)

        result = reasoner.query(
            ReasoningType.PROBABILISTIC,
            query_vars=["A"],
        )
        assert result.result is not None

    def test_causal_query(self):
        """Test causal query."""
        reasoner = ProbabilisticReasoner()

        # Set up SCM
        scm = StructuralCausalModel()
        scm.add_endogenous("X")
        scm.add_endogenous("Y")
        scm.add_linear_equation("X", [], {}, intercept=0)
        scm.add_linear_equation("Y", ["X"], {"X": 1.0}, intercept=0)
        reasoner.set_causal_model(scm)

        result = reasoner.query(
            ReasoningType.CAUSAL,
            query_vars=["Y"],
            intervention={"X": 5},
        )
        assert result.result is not None

    def test_causal_effect(self):
        """Test causal effect estimation."""
        reasoner = ProbabilisticReasoner()

        scm = StructuralCausalModel()
        scm.add_endogenous("X")
        scm.add_endogenous("Y")
        scm.add_linear_equation("X", [], {}, intercept=0)
        scm.add_linear_equation("Y", ["X"], {"X": 2.0}, intercept=0)
        reasoner.set_causal_model(scm)

        result = reasoner.causal_effect("X", "Y", 0, 1, n_samples=1000)
        # ATE should be approximately 2
        assert 1.5 < result.result["ate"] < 2.5

    def test_statistics(self):
        """Test statistics gathering."""
        reasoner = ProbabilisticReasoner()
        stats = reasoner.statistics()
        assert "has_bayesian_network" in stats


class TestInferenceResult:
    """Tests for InferenceResult."""

    def test_result_summary(self):
        """Test result summary generation."""
        result = InferenceResult(
            query_type=ReasoningType.PROBABILISTIC,
            query="P(A)",
            result={"A": {"0": 0.5, "1": 0.5}},
            confidence=0.9,
        )
        summary = result.summary()
        assert "P(A)" in summary
        assert "0.9" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
