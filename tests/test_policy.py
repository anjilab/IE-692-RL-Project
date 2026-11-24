"""Policy helpers and the action sampler shared by every estimator."""

import numpy as np
import pytest

from ie692_rl.core.policy import check_policy, sample_action, uniform_policy
from ie692_rl.types import FloatArray


def test_uniform_policy() -> None:
    policy = uniform_policy(5, 4)
    assert policy.shape == (5, 4)
    np.testing.assert_allclose(policy, 0.25)
    check_policy(policy, 5, 4)


@pytest.mark.parametrize(
    "policy",
    [
        np.full((3, 2), 0.5)[:2],  # wrong shape
        np.array([[0.5, 0.5], [1.2, -0.2], [0.5, 0.5]]),  # negative entry
        np.array([[0.5, 0.5], [np.nan, 1.0], [0.5, 0.5]]),  # NaN
        np.array([[0.5, 0.5], [np.inf, 0.0], [0.5, 0.5]]),  # inf
        np.array([[0.5, 0.4], [0.5, 0.5], [0.5, 0.5]]),  # row does not sum to 1
        np.zeros((3, 2)),  # all-zero rows
    ],
)
def test_check_policy_rejects(policy: FloatArray) -> None:
    with pytest.raises(ValueError, match=r"Policy shape|probabilities"):
        check_policy(policy, 3, 2)


def test_sample_action_matches_probabilities() -> None:
    probs = np.array([0.1, 0.0, 0.6, 0.3])
    cdf = np.cumsum(probs)
    rng, n = np.random.default_rng(0), 100_000
    draws = np.array([sample_action(cdf, probs, rng) for _ in range(n)])
    freq = np.bincount(draws, minlength=4) / n
    assert freq[1] == 0  # zero-probability actions are never drawn
    assert np.all(np.abs(freq - probs) < 5 * np.sqrt(probs * (1 - probs) / n) + 1e-12)


def test_sample_action_rounding_fallback() -> None:
    """If the cumulative row sums to less than the draw, the last positive action is returned."""
    probs = np.array([0.25, 0.25, 0.25, 0.0])
    cdf = np.cumsum(probs)  # ends at 0.75: draws above it must still return a valid action
    rng = np.random.default_rng(1)
    draws = np.array([sample_action(cdf, probs, rng) for _ in range(20_000)])
    assert set(np.unique(draws)) == {0, 1, 2}
    assert np.mean(draws == 2) == pytest.approx(0.5, abs=0.02)


def test_deterministic_row() -> None:
    probs = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(2)
    assert all(sample_action(np.cumsum(probs), probs, rng) == 2 for _ in range(100))
