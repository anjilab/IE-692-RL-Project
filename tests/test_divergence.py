"""Bregman divergence interface, KL (baseline + entropy enhancement) and the placeholder divergences.

``test_contract_*`` run against every registered divergence; placeholders that still raise
NotImplementedError are skipped, so a new implementation is checked automatically once filled in.
"""

import math

import numpy as np
import pytest

from ie692_rl.core.divergence import BregmanDivergence, negative_entropy
from ie692_rl.enhancements.bregman_spmd import KL, Euclidean, Tsallis, index_from_gamma, project_to_simplex
from ie692_rl.enhancements.bregman_spmd.tsallis import tsallis_step
from ie692_rl.experiment import DIVERGENCES
from ie692_rl.types import FloatArray
from reference import reference_tsallis_step

ALL = [
    KL(),
    Euclidean(),
    Tsallis(index_from_gamma(0.9)),
    Tsallis(0.5),
    Tsallis(1.5),
    Tsallis(2.0),
    Tsallis(3.5),
]


def simplex_rows(seed: int, rows: int = 6, actions: int = 4, zeros: bool = False) -> FloatArray:
    p: FloatArray = np.random.default_rng(seed).dirichlet(np.ones(actions), size=rows)
    if zeros:
        p[0] = [1.0] + [0.0] * (actions - 1)
        p[1, 1] = 0.0
        p[1] /= p[1].sum()
    return p


def implemented(divergence: BregmanDivergence) -> None:
    try:
        divergence.step(np.full((1, 2), 0.5), np.zeros((1, 2)), 1.0)
        divergence.divergence(np.full((1, 2), 0.5), np.full((1, 2), 0.5))
    except NotImplementedError:
        pytest.skip(f"{divergence!r} is a placeholder")


# ----------------------------------------------------------------------------- interface


def test_abstract_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BregmanDivergence()  # type: ignore[abstract]

    class Incomplete(BregmanDivergence):
        name = "incomplete"

        def divergence(self, p: FloatArray, center: FloatArray) -> FloatArray:
            return np.zeros(len(p))

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_registry_names_match_classes() -> None:
    assert set(DIVERGENCES) == {"kl", "euclidean", "tsallis"}
    for name, cls in DIVERGENCES.items():
        assert cls.name == name
        assert issubclass(cls, BregmanDivergence)


@pytest.mark.parametrize(
    ("policy", "q", "eta", "tau"),
    [
        (np.full((2, 2), 0.5), np.zeros((2, 3)), 1.0, 0.0),  # shape mismatch
        (np.full(2, 0.5), np.zeros(2), 1.0, 0.0),  # not a matrix
        (np.full((2, 2), 0.5), np.array([[0.0, np.nan], [0.0, 0.0]]), 1.0, 0.0),
        (np.full((2, 2), 0.5), np.array([[0.0, np.inf], [0.0, 0.0]]), 1.0, 0.0),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), 0.0, 0.0),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), -1.0, 0.0),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), math.nan, 0.0),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), 1.0, -0.1),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), 1.0, math.inf),
        (np.full((2, 2), 0.5), np.zeros((2, 2)), 1.0, math.nan),
    ],
)
def test_step_input_validation(policy: FloatArray, q: FloatArray, eta: float, tau: float) -> None:
    with pytest.raises(ValueError):  # noqa: PT011
        KL().step(policy, q, eta, tau)


# ----------------------------------------------------------------------------- negative entropy


def test_negative_entropy_values_and_zeros() -> None:
    p = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    np.testing.assert_allclose(negative_entropy(p), [-math.log(4), 0.0, -math.log(2)])
    assert np.all(np.isfinite(negative_entropy(simplex_rows(0, zeros=True))))


def test_negative_entropy_bounds() -> None:
    h = negative_entropy(simplex_rows(1, rows=500))
    assert np.all(h <= 1e-12)
    assert np.all(h >= -math.log(4) - 1e-12)


# ----------------------------------------------------------------------------- contract for every divergence


@pytest.mark.parametrize("divergence", ALL, ids=repr)
def test_contract_divergence_is_zero_on_the_diagonal_and_nonnegative(divergence: BregmanDivergence) -> None:
    implemented(divergence)
    p, c = simplex_rows(2, rows=200), simplex_rows(3, rows=200)
    np.testing.assert_allclose(divergence.divergence(p, p), 0.0, atol=1e-12)
    assert np.all(divergence.divergence(p, c) >= -1e-12)


@pytest.mark.parametrize("divergence", ALL, ids=repr)
@pytest.mark.parametrize("eta", [1e-6, 0.3, 5.0, 1e3])
def test_contract_step_returns_a_policy(divergence: BregmanDivergence, eta: float) -> None:
    implemented(divergence)
    policy, q = simplex_rows(4, zeros=True), np.random.default_rng(5).normal(size=(6, 4)) * 3
    new = divergence.step(policy, q, eta)
    assert new.shape == policy.shape
    assert np.all(np.isfinite(new))
    assert np.all(new >= 0)
    np.testing.assert_allclose(new.sum(axis=1), 1.0)


@pytest.mark.parametrize("divergence", ALL, ids=repr)
@pytest.mark.parametrize("eta", [0.1, 2.0, 20.0])
def test_contract_step_minimises_the_subproblem(divergence: BregmanDivergence, eta: float) -> None:
    """No point of the simplex (Dirichlet samples and vertices) beats the step on its objective."""
    implemented(divergence)
    policy = simplex_rows(6, rows=5)
    q = np.random.default_rng(7).uniform(0, 3, size=(5, 4))
    best = divergence.objective(divergence.step(policy, q, eta), policy, q, eta)
    candidates = np.random.default_rng(8).dirichlet(np.ones(4) * 0.5, size=4000)
    for row in range(5):
        trial = np.repeat(policy[row : row + 1], len(candidates), axis=0)
        values = divergence.objective(candidates, trial, np.repeat(q[row : row + 1], len(candidates), 0), eta)
        assert best[row] <= values.min() + 1e-9


@pytest.mark.parametrize("divergence", ALL, ids=repr)
def test_contract_small_step_barely_moves(divergence: BregmanDivergence) -> None:
    implemented(divergence)
    policy, q = simplex_rows(9), np.random.default_rng(10).normal(size=(6, 4))
    # 1e-12: with q > 1 the curvature x^(q-2) is tiny near 0, so small entries move ~ eta / x^(q-2)
    np.testing.assert_allclose(divergence.step(policy, q, 1e-12), policy, atol=1e-7)


@pytest.mark.parametrize("divergence", ALL, ids=repr)
def test_contract_step_lowers_the_expected_cost(divergence: BregmanDivergence) -> None:
    implemented(divergence)
    policy, q = simplex_rows(11), np.random.default_rng(12).normal(size=(6, 4))
    new = divergence.step(policy, q, 1.0)
    assert np.all((new * q).sum(axis=1) <= (policy * q).sum(axis=1) + 1e-12)


# ----------------------------------------------------------------------------- extremes, every divergence


@pytest.mark.parametrize("divergence", ALL, ids=repr)
@pytest.mark.parametrize("eta", [1e200, 1e308, math.inf])
def test_contract_huge_step_is_greedy(divergence: BregmanDivergence, eta: float) -> None:
    policy = simplex_rows(30, rows=3)  # full support
    q = np.array([[3.0, 1.0, 2.0, 9.0], [0.0, 2.0, 2.0, 5.0], [1e308, -1e308, 0.0, 5.0]])
    with np.errstate(all="raise"):
        new = divergence.step(policy, q, eta)
    np.testing.assert_allclose(new, np.eye(4)[[1, 0, 1]], atol=1e-12)


@pytest.mark.parametrize("divergence", ALL, ids=repr)
def test_contract_extreme_inputs_stay_finite(divergence: BregmanDivergence) -> None:
    policy = np.array([[1e-300, 0.5, 0.5 - 1e-300], [1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]])
    q = np.array([[1e308, -1e308, 0.0], [1e-300, 0.0, -1e-300], [1e5, 1e5 + 1, 1e5 + 2]])
    for eta in (1e-12, 1.0, 1e6):
        with np.errstate(all="raise"):
            new = divergence.step(policy, q, eta)
        assert np.all(np.isfinite(new))
        assert np.all(new >= 0)
        np.testing.assert_allclose(new.sum(axis=1), 1.0)


@pytest.mark.parametrize("divergence", [Euclidean(), Tsallis(0.5), Tsallis(3.5)], ids=repr)
def test_entropy_regularization_is_kl_only(divergence: BregmanDivergence) -> None:
    with pytest.raises(NotImplementedError, match="closed form only for KL"):
        divergence.step(np.full((1, 2), 0.5), np.zeros((1, 2)), 1.0, tau=0.1)


# ----------------------------------------------------------------------------- Euclidean


def test_euclidean_divergence_formula() -> None:
    p, c = simplex_rows(31), simplex_rows(32)
    np.testing.assert_allclose(Euclidean().divergence(p, c), 0.5 * ((p - c) ** 2).sum(axis=-1))


def test_simplex_projection_known_values() -> None:
    v = np.array([[0.5, 0.5], [2.0, 0.0], [0.7, 0.6], [0.2, 0.2], [-1.0, 3.0]])
    expected = np.array([[0.5, 0.5], [1, 0], [0.55, 0.45], [0.5, 0.5], [0, 1]], dtype=float)
    np.testing.assert_allclose(project_to_simplex(v), expected)


def test_simplex_projection_is_the_closest_point() -> None:
    v = np.random.default_rng(33).normal(size=(50, 5)) * 2
    projected = project_to_simplex(v)
    candidates = np.random.default_rng(34).dirichlet(np.ones(5) * 0.3, size=5000)
    for row in range(50):
        best = ((projected[row] - v[row]) ** 2).sum()
        assert best <= ((candidates - v[row]) ** 2).sum(axis=1).min() + 1e-12


def test_euclidean_step_can_zero_and_revive_actions() -> None:
    policy = np.array([[0.5, 0.5, 0.0]])
    new = Euclidean().step(policy, np.array([[0.0, 2.0, -0.5]]), 1.0)
    np.testing.assert_allclose(new, [[0.5, 0.0, 0.5]])  # action 1 hits 0 exactly, action 2 comes back


# ----------------------------------------------------------------------------- Tsallis


def test_index_from_gamma_matches_pg_termination() -> None:
    assert index_from_gamma(0.9) == pytest.approx(1 / (1 - math.log(0.1) / math.log(2)))
    assert index_from_gamma(0.9) == pytest.approx(0.2314, abs=1e-4)
    for gamma in (0.0, 1.0):
        with pytest.raises(ValueError, match="gamma"):
            index_from_gamma(gamma)


def test_tsallis_two_is_euclidean() -> None:
    policy, q = simplex_rows(35, rows=200, zeros=True), np.random.default_rng(36).normal(size=(200, 4))
    for eta in (0.1, 0.7, 5.0):
        np.testing.assert_allclose(
            Tsallis(2.0).step(policy, q, eta), Euclidean().step(policy, q, eta), atol=1e-13
        )
    p, c = simplex_rows(37), simplex_rows(38)
    np.testing.assert_allclose(Tsallis(2.0).divergence(p, c), Euclidean().divergence(p, c), atol=1e-14)


@pytest.mark.parametrize("index", [1 - 1e-6, 1 + 1e-6])
def test_tsallis_approaches_kl(index: float) -> None:
    policy, q = simplex_rows(39, rows=100), np.random.default_rng(40).normal(size=(100, 4))
    np.testing.assert_allclose(Tsallis(index).step(policy, q, 0.8), KL().step(policy, q, 0.8), atol=1e-5)
    p = simplex_rows(41, rows=100)
    np.testing.assert_allclose(Tsallis(index).divergence(p, policy), KL().divergence(p, policy), rtol=1e-4)


@pytest.mark.parametrize("index", [index_from_gamma(0.9), 0.5, 0.9])
@pytest.mark.parametrize("eta", [0.1, 1.0, 10.0])
def test_tsallis_matches_reference_implementation(index: float, eta: float) -> None:
    """pg-termination's update (with its two bugs fixed, see tests/reference.py) agrees to its 1e-6
    tolerance on the multiplier, and the Newton solution is never worse on the objective."""
    policy, q = simplex_rows(42, rows=60), np.random.default_rng(43).uniform(0, 10, size=(60, 4))
    tsallis = Tsallis(index)
    ours, reference = tsallis.step(policy, q, eta), reference_tsallis_step(policy, q, eta, index)
    np.testing.assert_allclose(ours, reference, atol=1e-3)
    assert np.all(
        tsallis.objective(ours, policy, q, eta) <= tsallis.objective(reference, policy, q, eta) + 1e-12
    )


def test_tsallis_newton_converges_in_few_iterations() -> None:
    """Bracketed, warm-started Newton: at most ~10 iterations per state on average, to machine precision."""
    policy = simplex_rows(44, rows=500, actions=13)
    for index in (index_from_gamma(0.9), 0.5, 1.5, 3.5):
        for eta in (0.01, 1.0, 100.0):
            scaled = Tsallis.scaled_costs(np.random.default_rng(45).uniform(0, 10, size=(500, 13)), eta)
            new, iterations = tsallis_step(policy, scaled, index)
            np.testing.assert_allclose(new.sum(axis=1), 1.0)
            assert iterations / 500 <= 10, (index, eta, iterations / 500)


def test_tsallis_barrier_below_one_and_zeros_above_one() -> None:
    policy, q = np.array([[0.6, 0.4, 0.0]]), np.array([[0.0, 3.0, -5.0]])
    assert Tsallis(0.5).step(policy, q, 1.0)[0, 2] == 0.0  # q < 1: an excluded action stays excluded
    assert Tsallis(0.5).step(policy, q, 1.0)[0, 1] > 0.0  # and nothing reaches 0 exactly
    assert Tsallis(3.5).step(policy, q, 1.0)[0, 2] > 0.0  # q > 1: no barrier, the cheap action comes in
    assert Tsallis(3.5).step(np.full((1, 3), 1 / 3), np.array([[0.0, 0.0, 50.0]]), 1.0)[0, 2] == 0.0


def test_tsallis_divergence_known_values() -> None:
    p, c = np.array([[0.5, 0.5]]), np.array([[0.25, 0.75]])
    q = 0.5  # w(x) = -4 (sum sqrt(x) - 1), grad w(x) = -2 / sqrt(x)

    def w(x: FloatArray) -> float:
        return float(-4 * (np.sqrt(x).sum() - 1))

    expected = w(p) - w(c) - (-2 / np.sqrt(c[0])) @ (p[0] - c[0])
    assert Tsallis(q).divergence(p, c)[0] == pytest.approx(expected)
    assert Tsallis(q).divergence(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))[0] == math.inf
    assert Tsallis(q).divergence(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0]]))[0] == 0.0
    assert math.isfinite(Tsallis(3.5).divergence(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))[0])


@pytest.mark.parametrize("index", [0.0, -1.0, 1.0, math.inf, math.nan])
def test_tsallis_index_validation(index: float) -> None:
    with pytest.raises(ValueError, match="index"):
        Tsallis(index)


def test_reprs() -> None:
    assert repr(KL()) == "KL()"
    assert repr(Tsallis(2.5)) == "Tsallis(index=2.5)"


# ----------------------------------------------------------------------------- KL specifics


def test_kl_divergence_known_values() -> None:
    p, c = np.array([[0.5, 0.5]]), np.array([[0.25, 0.75]])
    expected = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    assert KL().divergence(p, c)[0] == pytest.approx(expected)
    assert KL().divergence(np.array([[1.0, 0.0]]), c)[0] == pytest.approx(math.log(4))  # 0 log 0 = 0
    assert KL().divergence(np.array([[0.5, 0.5]]), np.array([[1.0, 0.0]]))[0] == math.inf


@pytest.mark.parametrize("tau", [0.0, 0.01, 0.5, 10.0])
@pytest.mark.parametrize("eta", [0.1, 1.0, 10.0])
def test_kl_closed_form(eta: float, tau: float) -> None:
    """pi_{k+1} proportional to pi_k^(1/(1+eta tau)) exp(-eta Q / (1+eta tau)) (proposal Section 2.3)."""
    policy, q = simplex_rows(13), np.random.default_rng(14).uniform(0, 2, size=(6, 4))
    expected = policy ** (1 / (1 + eta * tau)) * np.exp(-eta * q / (1 + eta * tau))
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(KL().step(policy, q, eta, tau), expected, rtol=1e-12)


@pytest.mark.parametrize("tau", [0.05, 1.0])
def test_kl_with_entropy_minimises_the_regularized_subproblem(tau: float) -> None:
    policy, q = simplex_rows(15, rows=4), np.random.default_rng(16).uniform(0, 3, size=(4, 4))
    kl = KL()
    best = kl.objective(kl.step(policy, q, 2.0, tau), policy, q, 2.0, tau)
    candidates = np.random.default_rng(17).dirichlet(np.ones(4), size=4000)
    for row in range(4):
        values = kl.objective(
            candidates,
            np.repeat(policy[row : row + 1], 4000, 0),
            np.repeat(q[row : row + 1], 4000, 0),
            2.0,
            tau,
        )
        assert best[row] <= values.min() + 1e-9


def test_kl_is_invariant_to_row_shifts_of_q() -> None:
    policy, q = simplex_rows(18), np.random.default_rng(19).normal(size=(6, 4))
    shift = np.random.default_rng(20).normal(size=(6, 1)) * 1e3
    np.testing.assert_allclose(
        KL().step(policy, q + shift, 3.0, 0.1), KL().step(policy, q, 3.0, 0.1), atol=1e-9
    )


def test_kl_keeps_zero_probabilities_at_zero() -> None:
    policy = simplex_rows(21, zeros=True)
    q = -np.ones_like(policy) * np.arange(4)  # the zero-probability actions are the cheapest
    new = KL().step(policy, q, 50.0, 0.3)
    assert np.all(new[policy == 0] == 0)


@pytest.mark.parametrize("eta", [1e200, 1e308, math.inf])
def test_kl_huge_step_is_greedy_on_the_support(eta: float) -> None:
    policy = np.array([[0.2, 0.3, 0.5, 0.0], [0.0, 0.5, 0.5, 0.0], [0.25, 0.25, 0.25, 0.25]])
    q = np.array([[3.0, 1.0, 2.0, -9.0], [0.0, 2.0, 2.0, 0.0], [1e308, -1e308, 0.0, 5.0]])
    with np.errstate(all="raise"):
        new = KL().step(policy, q, eta)
    np.testing.assert_allclose(new, np.array([[0, 1, 0, 0], [0, 0.5, 0.5, 0], [0, 1, 0, 0]], dtype=float))


def test_kl_infinite_step_with_entropy_is_boltzmann() -> None:
    policy = np.array([[0.1, 0.2, 0.7], [0.5, 0.5, 0.0]])
    q, tau = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, -5.0]]), 0.5
    expected = np.where(policy > 0, np.exp(-q / tau), 0.0)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(KL().step(policy, q, math.inf, tau), expected)


def test_kl_extreme_q_values_stay_finite() -> None:
    policy = np.full((3, 3), 1 / 3)
    q = np.array([[1e308, -1e308, 0.0], [1e-300, 0.0, -1e-300], [1e5, 1e5 + 1, 1e5 + 2]])
    with np.errstate(all="raise"):
        new = KL().step(policy, q, 100.0, 1e6)
    assert np.all(np.isfinite(new))
    np.testing.assert_allclose(new.sum(axis=1), 1.0)
