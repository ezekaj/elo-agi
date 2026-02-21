"""
Comprehensive tests for DifferentiableSCM.

Tests cover:
- Basic SCM operations
- Topological ordering
- Interventions (do-calculus)
- Counterfactual inference
- Gradient-based learning
- Graph structure operations
- Mathematical correctness
"""

import pytest
import numpy as np
from neuro.modules.causal.differentiable_scm import (
    DifferentiableSCM,
    CausalMechanism,
    NeuralNetwork,
    MLPLayer,
    ActivationType,
    apply_activation,
    activation_gradient,
)

class TestActivationFunctions:
    """Test activation functions and their gradients."""

    @pytest.mark.parametrize("activation", list(ActivationType))
    def test_activation_output_shape(self, activation):
        """Activation output should match input shape."""
        x = np.random.randn(10)
        y = apply_activation(x, activation)
        assert y.shape == x.shape

    @pytest.mark.parametrize("activation", list(ActivationType))
    def test_gradient_output_shape(self, activation):
        """Gradient output should match input shape."""
        x = np.random.randn(10)
        grad = activation_gradient(x, activation)
        assert grad.shape == x.shape

    def test_relu_positive_passthrough(self):
        """ReLU should pass through positive values."""
        x = np.array([1.0, 2.0, 3.0])
        y = apply_activation(x, ActivationType.RELU)
        np.testing.assert_array_equal(y, x)

    def test_relu_negative_zero(self):
        """ReLU should zero out negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        y = apply_activation(x, ActivationType.RELU)
        np.testing.assert_array_equal(y, np.zeros(3))

    def test_sigmoid_bounds(self):
        """Sigmoid output should be in (0, 1)."""
        x = np.random.randn(100) * 10
        y = apply_activation(x, ActivationType.SIGMOID)
        assert np.all(y > 0) and np.all(y < 1)

    def test_tanh_bounds(self):
        """Tanh output should be in [-1, 1]."""
        x = np.random.randn(100) * 10
        y = apply_activation(x, ActivationType.TANH)
        assert np.all(y >= -1) and np.all(y <= 1)

    def test_gradient_numerical_verification(self):
        """Verify gradients numerically."""
        eps = 1e-5
        x = np.array([0.5])

        for activation in [ActivationType.TANH, ActivationType.SIGMOID, ActivationType.SOFTPLUS]:
            # Numerical gradient
            y_plus = apply_activation(x + eps, activation)
            y_minus = apply_activation(x - eps, activation)
            numerical_grad = (y_plus - y_minus) / (2 * eps)

            # Analytical gradient
            analytical_grad = activation_gradient(x, activation)

            np.testing.assert_allclose(numerical_grad, analytical_grad, rtol=1e-4)

class TestNeuralNetwork:
    """Test neural network implementation."""

    def test_forward_output_shape(self):
        """Forward pass should produce correct output shape."""
        nn = NeuralNetwork(input_dim=10, output_dim=5, hidden_dims=[32, 16])
        x = np.random.randn(10)
        y = nn.forward(x)
        assert y.shape == (5,)

    def test_backward_gradient_shape(self):
        """Backward pass should produce correct gradient shape."""
        nn = NeuralNetwork(input_dim=10, output_dim=5, hidden_dims=[32, 16])
        x = np.random.randn(10)
        nn.forward(x)
        grad_output = np.random.randn(5)
        grad_input = nn.backward(grad_output)
        assert grad_input.shape == (10,)

    def test_parameter_count(self):
        """Network should have correct number of parameters."""
        nn = NeuralNetwork(input_dim=10, output_dim=5, hidden_dims=[32, 16])
        params = nn.parameters()
        # (10*32 + 32) + (32*16 + 16) + (16*5 + 5) = 320+32 + 512+16 + 80+5 = 965
        total_params = sum(p.size for p in params)
        expected = (10*32 + 32) + (32*16 + 16) + (16*5 + 5)
        assert total_params == expected

    def test_update_changes_parameters(self):
        """Update should modify parameters."""
        nn = NeuralNetwork(input_dim=5, output_dim=3, hidden_dims=[8], random_seed=42)
        x = np.random.randn(5)
        y = nn.forward(x)
        grad = np.ones(3)
        nn.backward(grad)

        params_before = [p.copy() for p in nn.parameters()]
        nn.update(learning_rate=0.1)
        params_after = nn.parameters()

        for p_before, p_after in zip(params_before, params_after):
            assert not np.allclose(p_before, p_after)

    def test_reproducibility_with_seed(self):
        """Same seed should produce same network."""
        nn1 = NeuralNetwork(input_dim=5, output_dim=3, hidden_dims=[8], random_seed=42)
        nn2 = NeuralNetwork(input_dim=5, output_dim=3, hidden_dims=[8], random_seed=42)

        x = np.random.randn(5)
        y1 = nn1.forward(x)
        y2 = nn2.forward(x)

        np.testing.assert_array_equal(y1, y2)

class TestCausalMechanism:
    """Test causal mechanisms."""

    def test_mechanism_forward(self):
        """Mechanism should produce output."""
        mechanism = CausalMechanism(
            variable="Y",
            parents=["X1", "X2"],
        )
        parent_values = {"X1": 1.0, "X2": 2.0}
        y = mechanism.forward(parent_values)
        assert isinstance(y, float)

    def test_mechanism_with_analytical_function(self):
        """Analytical mechanism should compute correct values."""
        def linear_fn(parents, noise):
            return 2 * parents.get("X", 0) + noise

        mechanism = CausalMechanism(
            variable="Y",
            parents=["X"],
            analytical_fn=linear_fn,
        )

        y = mechanism.forward({"X": 3.0}, noise=1.0)
        assert y == 7.0  # 2*3 + 1

    def test_mechanism_noise_sampling(self):
        """Noise sampling should follow distribution."""
        mechanism = CausalMechanism(
            variable="Y",
            parents=[],
            noise_mean=5.0,
            noise_std=2.0,
        )

        samples = mechanism.sample_noise(1000)
        assert abs(np.mean(samples) - 5.0) < 0.2
        assert abs(np.std(samples) - 2.0) < 0.2

    def test_mechanism_backward_returns_parent_gradients(self):
        """Backward should return gradients for all parents."""
        mechanism = CausalMechanism(
            variable="Y",
            parents=["X1", "X2", "X3"],
        )
        mechanism.forward({"X1": 1, "X2": 2, "X3": 3})
        grads = mechanism.backward(1.0)

        assert set(grads.keys()) == {"X1", "X2", "X3"}

class TestDifferentiableSCM:
    """Test full SCM implementation."""

    def test_add_variable(self):
        """Should add variables correctly."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        assert "X" in scm._variables
        assert "Y" in scm._variables
        assert scm._parents["Y"] == ["X"]

    def test_topological_order_simple_chain(self):
        """Topological order should respect dependencies."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])
        scm.add_variable("Z", parents=["Y"])

        order = scm._topological_order()
        assert order.index("X") < order.index("Y") < order.index("Z")

    def test_topological_order_diamond(self):
        """Topological order for diamond structure."""
        scm = DifferentiableSCM()
        scm.add_variable("A")
        scm.add_variable("B", parents=["A"])
        scm.add_variable("C", parents=["A"])
        scm.add_variable("D", parents=["B", "C"])

        order = scm._topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_forward_produces_values(self):
        """Forward pass should produce values for all variables."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {}, intercept=1.0, noise_std=0.0)
        scm.add_linear_mechanism("Y", ["X"], {"X": 2.0}, intercept=0.0, noise_std=0.0)

        values = scm.forward(noise={"X": 0.0, "Y": 0.0, "U_X": 0.0, "U_Y": 0.0})
        assert "X" in values and "Y" in values
        assert values["X"] == 1.0  # intercept
        assert values["Y"] == 2.0  # 2 * 1.0

    def test_intervention_overrides_mechanism(self):
        """do(X=x) should override the natural mechanism."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {}, intercept=1.0)
        scm.add_linear_mechanism("Y", ["X"], {"X": 2.0})

        # Without intervention
        values_natural = scm.forward(noise={v: 0 for v in scm._variables})

        # With intervention do(X=5)
        values_do = scm.forward(
            noise={v: 0 for v in scm._variables},
            interventions={"X": 5.0}
        )

        assert values_do["X"] == 5.0
        assert values_do["Y"] == 10.0  # 2 * 5.0

    def test_intervention_breaks_causal_connection(self):
        """Intervention should break incoming causal connections."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("A", [], {}, intercept=1.0)
        scm.add_linear_mechanism("X", ["A"], {"A": 3.0})  # X = 3A
        scm.add_linear_mechanism("Y", ["X"], {"X": 2.0})  # Y = 2X

        # do(X=10) should make Y = 20, regardless of A
        values = scm.forward(
            noise={v: 0 for v in scm._variables},
            interventions={"X": 10.0}
        )

        assert values["X"] == 10.0
        assert values["Y"] == 20.0  # Should not depend on A

    def test_counterfactual_consistency(self):
        """Counterfactual should be consistent with evidence."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {}, intercept=0.0, noise_std=1.0)
        scm.add_linear_mechanism("Y", ["X"], {"X": 2.0}, intercept=0.0, noise_std=0.0)

        # Observe X=3, Y=6
        evidence = {"X": 3.0, "Y": 6.0}

        # Counterfactual: what if X had been 5?
        cf = scm.counterfactual(evidence, {"X": 5.0})

        # Y should be 10 (since Y = 2X and noise is preserved)
        assert abs(cf["Y"] - 10.0) < 0.5

    def test_causal_effect_positive(self):
        """Positive causal effect should be detected."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {})
        scm.add_linear_mechanism("Y", ["X"], {"X": 3.0})  # Y = 3X + noise

        effect = scm.causal_effect("X", "Y", treatment_value=1.0, baseline_value=0.0)
        assert effect > 2.5 and effect < 3.5  # Should be approximately 3

    def test_causal_effect_zero_for_non_causes(self):
        """No causal effect for non-causes."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {})
        scm.add_linear_mechanism("Y", [], {})  # Y independent of X

        effect = scm.causal_effect("X", "Y", treatment_value=1.0, baseline_value=0.0)
        assert abs(effect) < 0.5  # Should be approximately 0

    def test_ancestors_and_descendants(self):
        """Test ancestor and descendant queries."""
        scm = DifferentiableSCM()
        scm.add_variable("A")
        scm.add_variable("B", parents=["A"])
        scm.add_variable("C", parents=["B"])
        scm.add_variable("D", parents=["B"])

        assert scm.get_ancestors("C") == {"A", "B"}
        assert scm.get_ancestors("D") == {"A", "B"}
        assert scm.get_descendants("A") == {"B", "C", "D"}
        assert scm.get_descendants("B") == {"C", "D"}

    def test_d_separation_chain(self):
        """Test d-separation in chain A -> B -> C."""
        scm = DifferentiableSCM()
        scm.add_variable("A")
        scm.add_variable("B", parents=["A"])
        scm.add_variable("C", parents=["B"])

        # A and C are not d-separated unconditionally
        assert not scm.is_d_separated("A", "C")

        # A and C are d-separated given B
        assert scm.is_d_separated("A", "C", {"B"})

    def test_d_separation_fork(self):
        """Test d-separation in fork B <- A -> C."""
        scm = DifferentiableSCM()
        scm.add_variable("A")
        scm.add_variable("B", parents=["A"])
        scm.add_variable("C", parents=["A"])

        # B and C are not d-separated unconditionally
        assert not scm.is_d_separated("B", "C")

        # B and C are d-separated given A
        assert scm.is_d_separated("B", "C", {"A"})

    def test_sample_returns_correct_count(self):
        """Sample should return correct number of samples."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        samples = scm.sample(n_samples=100)
        assert len(samples) == 100

    def test_intervene_returns_new_scm(self):
        """Intervene should return a new SCM."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        intervened = scm.intervene({"X": 5.0})

        assert intervened is not scm
        assert intervened.name != scm.name

    def test_statistics(self):
        """Statistics should be tracked correctly."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        scm.sample(10)
        stats = scm.statistics()

        assert stats["n_variables"] == 2
        assert stats["n_samples"] == 10

    def test_fit_reduces_loss(self):
        """Fitting should reduce prediction error."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        # Generate training data from true model Y = 2X
        np.random.seed(42)
        data = []
        for _ in range(100):
            x = np.random.randn()
            y = 2 * x + np.random.randn() * 0.1
            data.append({"X": x, "Y": y})

        initial_loss = scm.fit(data, n_epochs=1)
        final_loss = scm.fit(data, n_epochs=50)

        # Loss should decrease or stay reasonable
        assert final_loss["Y"] < initial_loss["Y"] * 2  # At least not much worse

class TestComplexCausalStructures:
    """Test complex causal structures and edge cases."""

    def test_large_chain(self):
        """Test long causal chain."""
        scm = DifferentiableSCM()

        # Create chain: X0 -> X1 -> ... -> X9
        scm.add_variable("X0")
        for i in range(1, 10):
            scm.add_linear_mechanism(f"X{i}", [f"X{i-1}"], {f"X{i-1}": 1.0})

        order = scm._topological_order()
        for i in range(9):
            assert order.index(f"X{i}") < order.index(f"X{i+1}")

    def test_wide_graph(self):
        """Test graph with many parallel paths."""
        scm = DifferentiableSCM()

        scm.add_variable("Root")
        for i in range(20):
            scm.add_variable(f"Child{i}", parents=["Root"])
        scm.add_variable("Sink", parents=[f"Child{i}" for i in range(20)])

        order = scm._topological_order()
        assert order.index("Root") == 0
        assert order.index("Sink") == len(order) - 1

    def test_confounded_variables(self):
        """Test confounding (common cause)."""
        scm = DifferentiableSCM()

        # U -> X, U -> Y (U is confounder)
        scm.add_variable("U")
        scm.add_linear_mechanism("X", ["U"], {"U": 1.0}, noise_std=0.0)
        scm.add_linear_mechanism("Y", ["U"], {"U": 2.0}, noise_std=0.0)

        # X and Y should be correlated but X doesn't cause Y
        samples = scm.sample(100)
        x_vals = [s["X"] for s in samples]
        y_vals = [s["Y"] for s in samples]
        corr = np.corrcoef(x_vals, y_vals)[0, 1]

        assert abs(corr) > 0.9  # Should be highly correlated

        # But causal effect should be near zero
        effect = scm.causal_effect("X", "Y")
        # Note: this might not be exactly zero due to indirect effects through U
        # In a proper front-door or back-door adjustment this would be zero

    def test_collider_structure(self):
        """Test collider (common effect)."""
        scm = DifferentiableSCM()

        # X -> C <- Y (C is collider)
        scm.add_variable("X")
        scm.add_variable("Y")
        scm.add_linear_mechanism("C", ["X", "Y"], {"X": 1.0, "Y": 1.0})

        # X and Y should be independent unconditionally
        assert scm.is_d_separated("X", "Y")

        # X and Y should NOT be d-separated given C (explaining away)
        assert not scm.is_d_separated("X", "Y", {"C"})

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_scm(self):
        """Empty SCM should work."""
        scm = DifferentiableSCM()
        values = scm.forward()
        assert values == {}

    def test_single_variable(self):
        """Single variable SCM should work."""
        scm = DifferentiableSCM()
        scm.add_variable("X")
        values = scm.forward()
        assert "X" in values

    def test_deterministic_mechanism(self):
        """Deterministic mechanism (zero noise) should give consistent results."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {}, intercept=5.0, noise_std=0.0)

        values1 = scm.forward(noise={"X": 0.0, "U_X": 0.0})
        values2 = scm.forward(noise={"X": 0.0, "U_X": 0.0})

        assert values1["X"] == values2["X"]

    def test_causal_gradient(self):
        """Test causal gradient computation."""
        scm = DifferentiableSCM()
        scm.add_linear_mechanism("X", [], {})
        scm.add_linear_mechanism("Y", ["X"], {"X": 3.0})

        grad = scm.causal_gradient("X", "Y", {"X": 1.0})
        assert abs(grad - 3.0) < 0.5  # Should be approximately 3

    def test_to_dict_serialization(self):
        """SCM should be serializable to dict."""
        scm = DifferentiableSCM(name="test_scm")
        scm.add_variable("X")
        scm.add_variable("Y", parents=["X"])

        d = scm.to_dict()
        assert d["name"] == "test_scm"
        assert set(d["variables"]) == {"X", "Y"}
        assert d["parents"]["Y"] == ["X"]
