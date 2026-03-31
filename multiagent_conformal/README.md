# Conformal Navigation: Multi-Agent Uncertainty Demo

A prototype demonstrating **conformal prediction** applied to **multi-agent navigation** under uncertainty.

## What It Does

Two agents navigate a 10x10 grid from opposite corners toward their goals. Neither agent can see the other directly -- they only receive **noisy observations**. Each agent uses **conformal prediction** to build a statistically guaranteed uncertainty set about the other's position, then uses a **minimax policy** to choose safe actions.

**Key result:** 0 collisions across 50 timesteps, with empirical coverage consistently above the 90% target.

## The Pipeline

```
Noisy Observation
      ↓
Logistic Regression Classifier
(trained on 20,000 obs → true position pairs)
      ↓
Conformity Scores on Calibration Set
(4,000 held-out examples)
      ↓
Threshold q = Quantile(scores, α)
      ↓
Prediction Set C = { i : P(i | obs) ≥ q }
(guaranteed: P(true pos ∈ C) ≥ 1 − α)
      ↓
Minimax Policy: a* = argmin_a [ max_{p ∈ C} cost(a, p) ]
      ↓
Agent moves safely
```

## Key Concepts

- **Conformal Prediction:** A framework that wraps any classifier and produces prediction sets with a provable coverage guarantee. Not a heuristic -- a mathematical theorem under exchangeability.
- **Conformity Score:** The probability the classifier assigns to the true label on calibration examples. Used to empirically calibrate the threshold q.
- **Minimax Policy:** Plans for the worst-case position in the uncertainty set. Conservative but safe.
- **Coverage Guarantee:** P(y_test ∈ C(obs)) ≥ 1 − α, proven under exchangeability of calibration and test data.

## Project Structure

```
multiagent_conformal/
├── environment.py   # Layer 1: 10x10 grid, agents, Gaussian noise model
├── classifier.py    # Layer 2: dataset generation, logistic regression training
├── conformal.py     # Layer 3: conformity scores, threshold q, prediction sets
├── policy.py        # Layer 4: minimax policy, full simulation
├── visualize.py     # Layer 5: matplotlib animation (offline)
├── index.html       # Web app: interactive browser demo (no server needed)
└── README.md
```

## Running the Python Version

```bash
pip install numpy scikit-learn matplotlib

python classifier.py   # trains model, saves model.pkl
python conformal.py    # calibrates predictor, updates model.pkl
python policy.py       # runs simulation, prints results
python visualize.py    # generates simulation.gif + simulation_final.png
```

Each layer builds on the previous. Run them in order.

## Running the Web App

Just open `index.html` in any browser. No server, no dependencies, no install.

The web app re-implements the full pipeline in JavaScript:
- Logistic regression with SGD
- Softmax over 100 classes
- Conformal calibration
- Minimax policy

Use the **α slider** to change the coverage target and see how prediction set sizes change. Use **Speed** to control playback rate. Hit **Reset** to retrain with the new α.

## The α Tradeoff

| α | Coverage Target | Avg Set Size |
|---|---|---|
| 0.30 | 70% | ~12 cells |
| 0.20 | 80% | ~15 cells |
| 0.10 | 90% | ~22 cells |
| 0.05 | 95% | ~27 cells |

Lower α = larger sets = stronger guarantee. The right α depends on the application's tolerance for risk.

## Author

Aly Ahmed -- UW CS + Math, Class of 2028  
ahatim@uw.edu
