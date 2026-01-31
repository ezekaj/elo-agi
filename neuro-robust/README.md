# neuro-robust

Robustness, uncertainty, and safety for neural systems.

## Overview

neuro-robust provides robustness capabilities including:

- **Uncertainty Quantification**: MC Dropout, ensembles, evidential methods
- **Out-of-Distribution Detection**: Mahalanobis, energy-based, isolation forest
- **Confidence Calibration**: Temperature scaling, Platt, isotonic regression
- **Adversarial Defense**: FGSM, PGD attacks and defenses
- **Selective Prediction**: Abstention policies for safe inference

## Installation

```bash
cd neuro-robust
pip install -r requirements.txt
```

## Quick Start

```python
from src.uncertainty import UncertaintyQuantifier, MCDropout
from src.ood_detection import OODDetector, MahalanobisDetector
from src.calibration import ConfidenceCalibrator, TemperatureScaling
from src.robust_inference import RobustInference

# Uncertainty quantification
uq = UncertaintyQuantifier(method="mc_dropout", n_samples=50)
prediction, uncertainty = uq.predict_with_uncertainty(model, input)

# OOD detection
ood = OODDetector(method="mahalanobis")
ood.fit(train_features)
is_ood = ood.detect(test_input)
ood_score = ood.score(test_input)

# Confidence calibration
calibrator = ConfidenceCalibrator(method="temperature")
calibrator.fit(logits, labels)
calibrated_probs = calibrator.calibrate(new_logits)

# Robust inference with abstention
robust = RobustInference(
    uncertainty_threshold=0.3,
    ood_threshold=0.8,
    abstain_policy="conservative"
)
result = robust.predict(model, input)
if result.abstained:
    print("Model uncertain, abstaining from prediction")
```

## Components

### UncertaintyQuantifier

Multiple uncertainty estimation methods:
- **MC Dropout**: Dropout at inference for Bayesian approximation
- **Deep Ensembles**: Multiple model predictions
- **Evidential**: Direct uncertainty from Dirichlet parameters
- Separates aleatoric and epistemic uncertainty

### OODDetector

Out-of-distribution detection:
- **Mahalanobis Distance**: Feature-space distance from training
- **Energy Score**: Negative log-sum-exp of logits
- **Isolation Forest**: Anomaly detection in feature space
- Returns calibrated OOD scores

### ConfidenceCalibrator

Calibrate model confidence to match accuracy:
- **Temperature Scaling**: Learn single temperature parameter
- **Platt Scaling**: Logistic regression on logits
- **Isotonic Regression**: Non-parametric calibration
- **Histogram Binning**: Bin-wise probability adjustment
- Measures ECE (Expected Calibration Error)

### AdversarialDefense

Attack and defense methods:
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **C&W**: Carlini-Wagner attack
- Adversarial training for robust models

### RobustInference

Safe prediction with rejection:
- Combine uncertainty, OOD, and calibration
- Configurable abstention policies
- Risk-aware decision making
- Graceful degradation under uncertainty

## API Reference

See [docs/api.md](docs/api.md) for detailed documentation.

## Theory

The module implements:
- Bayesian Deep Learning (Gal & Ghahramani)
- Calibration Theory (Guo et al.)
- Adversarial Examples (Goodfellow et al.)
- Selective Prediction (El-Yaniv & Wiener)

See [docs/theory.md](docs/theory.md) for theoretical background.

## Tests

```bash
python -m pytest tests/ -v
```

192 tests covering:
- Uncertainty estimation accuracy
- OOD detection AUROC
- Calibration error metrics
- Adversarial robustness

## Integration

Integrates with SharedSpace for robust embeddings:

```python
from src.integration import SharedSpaceRobustness

robustness = SharedSpaceRobustness(shared_space)
robust_embedding = robustness.add_uncertainty_to_embedding(embedding, input)
print(f"Uncertainty: {robust_embedding.uncertainty}")
print(f"OOD Score: {robust_embedding.ood_score}")
```

## License

MIT
