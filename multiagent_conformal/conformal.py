import numpy as np
import pickle
from classifier import pos_to_idx, idx_to_pos, compute_conformity_scores, GRID_SIZE

# -------------------------------------------------------
# CONFORMAL PREDICTOR CLASS
# -------------------------------------------------------
class ConformalPredictor:
    """
    Wraps a trained classifier and adds conformal prediction.

    This class is the core of Layer 3. It takes any trained
    classifier that outputs probabilities, uses a calibration
    set to find the threshold q, and then produces prediction
    sets with a guaranteed coverage level at test time.

    Think of it as a layer sitting on top of the classifier:
    Classifier -> raw probabilities -> ConformalPredictor -> prediction set
    """

    def __init__(self, clf, alpha=0.10):
        """
        clf:   a trained scikit-learn classifier with predict_proba()
        alpha: the error rate. alpha=0.10 means 90% coverage target.
               This is the ONE parameter you set upfront that controls
               the size vs. coverage tradeoff.
        """
        self.clf = clf
        self.alpha = alpha
        self.q = None  # threshold, set during calibration

    def calibrate(self, X_calib, y_calib):
        """
        Computes the threshold q from the calibration set.

        This is the step that converts the classifier's raw
        probabilities into a guaranteed coverage level.

        The math:
            q = Quantile(conformity_scores, alpha)

        With the finite sample correction:
            q = Quantile(scores, ceil((n+1) * alpha) / n)

        The (n+1) correction makes the guarantee exact rather
        than approximate. Without it, the coverage guarantee
        holds asymptotically (for large n) but may slightly
        undercover for small calibration sets.

        Args:
            X_calib: calibration observations, shape (n, 2)
            y_calib: true class indices, shape (n,)
        """
        # Step 1: get conformity scores from calibration set.
        # Each score = probability the classifier gave to the true label.
        # Imported from classifier.py -- same function we built in Layer 2.
        scores = compute_conformity_scores(self.clf, X_calib, y_calib)

        n = len(scores)

        # Step 2: compute q using the finite sample correction.
        # np.ceil() rounds up to the nearest integer.
        # np.quantile(arr, p) returns the p-th quantile of arr.
        # We divide by n (not n+1) because np.quantile interpolates
        # within the existing scores rather than extrapolating beyond them.
        corrected_quantile = np.ceil((n + 1) * self.alpha) / n

        # Clip to [0, 1] in case of floating point edge cases
        corrected_quantile = np.clip(corrected_quantile, 0, 1)

        self.q = np.quantile(scores, corrected_quantile)

        return scores  # return scores so caller can inspect them

    def predict_set(self, obs):
        """
        Produces a prediction set for a single observation.

        Given a noisy observation (obs_x, obs_y), returns the
        set of all grid positions whose probability exceeds q.

        The guarantee: the true position is in this set with
        probability at least (1 - alpha).

        Args:
            obs: (obs_x, obs_y) tuple -- the noisy observation

        Returns:
            prediction_set: list of (x, y) positions -- all positions
                            included in the conformal prediction set
            probs:          full probability array over all 100 cells
                            (useful for visualization and debugging)
        """
        if self.q is None:
            raise ValueError("Must call calibrate() before predict_set()")

        # Reshape obs to (1, 2) because predict_proba expects a 2D array.
        # np.array([obs]) creates shape (1, 2): one sample, two features.
        obs_array = np.array([obs])

        # predict_proba returns shape (1, 100): one row, 100 class probabilities.
        # [0] takes the first (only) row, giving us shape (100,).
        probs = self.clf.predict_proba(obs_array)[0]

        # Build the prediction set: include every class whose probability >= q.
        # np.where(condition) returns indices where condition is True.
        # probs >= self.q produces a boolean array of length 100.
        # np.where returns the indices where it's True.
        included_indices = np.where(probs >= self.q)[0]

        # Convert flat indices back to (x, y) positions using idx_to_pos.
        prediction_set = [idx_to_pos(idx) for idx in included_indices]

        return prediction_set, probs

    def predict_set_batch(self, X):
        """
        Produces prediction sets for a batch of observations.

        Same logic as predict_set() but vectorized for speed.
        Used during evaluation to process the entire test set at once.

        Args:
            X: array of shape (n_samples, 2) -- multiple observations

        Returns:
            prediction_sets: list of lists -- one prediction set per observation
            probs_batch:     array of shape (n_samples, 100) -- all probabilities
        """
        # Get all probabilities at once -- much faster than looping.
        probs_batch = self.clf.predict_proba(X)

        prediction_sets = []
        for probs in probs_batch:
            # Same threshold logic as predict_set(), applied row by row.
            included_indices = np.where(probs >= self.q)[0]
            prediction_set = [idx_to_pos(idx) for idx in included_indices]
            prediction_sets.append(prediction_set)

        return prediction_sets, probs_batch


# -------------------------------------------------------
# COVERAGE EVALUATION
# -------------------------------------------------------
def evaluate_coverage(predictor, X_test, y_test):
    """
    Measures the empirical coverage of the conformal predictor
    on the test set.

    Empirical coverage = fraction of test examples where the
    true label is inside the prediction set.

    This should be >= (1 - alpha). If it is, the guarantee holds.
    If it's significantly below (1 - alpha), something went wrong
    (usually distribution shift or a bug in the calibration step).

    Args:
        predictor: a calibrated ConformalPredictor
        X_test:    test observations, shape (n_test, 2)
        y_test:    true class indices, shape (n_test,)

    Returns:
        coverage:      fraction of test examples where true label is in set
        avg_set_size:  average number of positions in each prediction set
        set_sizes:     array of set sizes, one per test example
    """
    prediction_sets, _ = predictor.predict_set_batch(X_test)

    covered = 0      # count of examples where true label is in set
    set_sizes = []   # size of prediction set for each example

    for i, pred_set in enumerate(prediction_sets):
        true_pos = idx_to_pos(y_test[i])  # convert index back to (x,y)

        # Check if the true position is in the prediction set.
        # We use 'in' which checks for equality with each element.
        if true_pos in pred_set:
            covered += 1

        set_sizes.append(len(pred_set))

    set_sizes = np.array(set_sizes)
    coverage = covered / len(y_test)
    avg_set_size = set_sizes.mean()

    return coverage, avg_set_size, set_sizes


# -------------------------------------------------------
# MAIN: CALIBRATE AND EVALUATE
# -------------------------------------------------------
if __name__ == '__main__':
    print("=== Layer 3: Conformal Prediction ===\n")

    # Load everything saved by Layer 2
    print("Loading classifier and data splits from model.pkl...")
    with open('model.pkl', 'rb') as f:
        saved = pickle.load(f)

    clf     = saved['clf']
    X_calib = saved['X_calib']
    X_test  = saved['X_test']
    y_calib = saved['y_calib']
    y_test  = saved['y_test']
    print("  Loaded.\n")

    # -------------------------------------------------------
    # TEST MULTIPLE ALPHA LEVELS
    # -------------------------------------------------------
    # We evaluate at several alpha levels to show the tradeoff:
    # lower alpha = higher coverage guarantee = larger prediction sets
    alphas = [0.30, 0.20, 0.10, 0.05]

    print(f"{'Coverage Target':>20} {'Alpha':>8} {'q':>10} {'Actual Coverage':>18} {'Avg Set Size':>15}")
    print("-" * 75)

    for alpha in alphas:
        # Create and calibrate a predictor at this alpha level
        predictor = ConformalPredictor(clf, alpha=alpha)
        predictor.calibrate(X_calib, y_calib)

        # Evaluate on test set
        coverage, avg_set_size, set_sizes = evaluate_coverage(
            predictor, X_test, y_test
        )

        target = 1 - alpha
        print(f"{target*100:>19.0f}% {alpha:>8.2f} {predictor.q:>10.4f} "
              f"{coverage*100:>17.1f}% {avg_set_size:>15.1f}")

    print()

    # -------------------------------------------------------
    # DEEP DIVE AT 90% COVERAGE
    # -------------------------------------------------------
    print("Deep dive at 90% coverage (alpha=0.10):\n")

    predictor_90 = ConformalPredictor(clf, alpha=0.10)
    predictor_90.calibrate(X_calib, y_calib)

    coverage, avg_set_size, set_sizes = evaluate_coverage(
        predictor_90, X_test, y_test
    )

    print(f"  Threshold q:         {predictor_90.q:.4f}")
    print(f"  Empirical coverage:  {coverage*100:.1f}%  (target: 90%)")
    print(f"  Average set size:    {avg_set_size:.1f} cells out of 100")
    print(f"  Min set size:        {set_sizes.min()} cells")
    print(f"  Max set size:        {set_sizes.max()} cells")
    print()

    # Distribution of set sizes
    print("  Set size distribution:")
    for size in sorted(np.unique(set_sizes)):
        count = np.sum(set_sizes == size)
        pct = count / len(set_sizes) * 100
        bar = '#' * int(pct / 2)
        print(f"    Size {size:>3}: {count:>5} examples ({pct:>5.1f}%)  {bar}")
    print()

    # -------------------------------------------------------
    # SHOW A CONCRETE EXAMPLE
    # -------------------------------------------------------
    print("Concrete example -- single observation:\n")

    # Pick one test example
    example_obs = X_test[0]
    example_true = idx_to_pos(y_test[0])

    pred_set, probs = predictor_90.predict_set(tuple(example_obs.astype(int)))

    print(f"  Observation (what agent sees): {tuple(example_obs.astype(int))}")
    print(f"  True position (unknown to agent): {example_true}")
    print(f"  Prediction set ({len(pred_set)} cells): {sorted(pred_set)}")
    print(f"  True position in set: {example_true in pred_set}")
    print()

    # Show top 5 probabilities
    print("  Top 5 class probabilities:")
    top5_idx = np.argsort(probs)[::-1][:5]
    for idx in top5_idx:
        pos = idx_to_pos(idx)
        in_set = probs[idx] >= predictor_90.q
        marker = "<-- in set" if in_set else ""
        print(f"    {pos}: {probs[idx]:.4f}  {marker}")
    print()

    # Save calibrated predictor for use in later layers
    saved['predictor_90'] = predictor_90
    with open('model.pkl', 'wb') as f:
        pickle.dump(saved, f)
    print("  Saved calibrated predictor to model.pkl")
    print("\nLayer 3 complete.")
