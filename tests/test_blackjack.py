"""Simplified Blackjack: encoding, deal distribution, dealer rules, model and simulator agreement."""

import itertools

import numpy as np
import pytest

from ie692_rl.core.exact import ExactScorer
from ie692_rl.envs.blackjack import DRAW, STOP, Blackjack, dealer_final_distribution

ENV = Blackjack()


def test_sizes_and_settings() -> None:
    assert (ENV.n_states, ENV.n_actions, ENV.gamma) == (32, 2, 0.9)
    assert ENV.decode(ENV.start) == (8, 3)
    assert ENV.action_names == ("stop", "draw")
    with pytest.raises(ValueError, match="gamma"):
        Blackjack(gamma=1.0)


def test_encode_decode_is_a_bijection() -> None:
    states = {ENV.encode(t, d) for t in range(4, 12) for d in range(2, 6)}
    assert states == set(range(32))
    for s in range(32):
        assert ENV.encode(*ENV.decode(s)) == s
        assert ENV.state_label(s) == dict(zip(("player_total", "dealer_card"), ENV.decode(s), strict=True))


def test_deal_distribution() -> None:
    """Player total = sum of two uniform cards 2..5; the dealer's card is uniform and independent."""
    n = 200_000
    deals = ENV.deal(n, np.random.default_rng(0))
    totals, dealers = zip(*(ENV.decode(int(s)) for s in deals), strict=True)
    two_cards = np.bincount([a + b for a, b in itertools.product(range(2, 6), repeat=2)], minlength=12) / 16
    freq = np.bincount(totals, minlength=12) / n
    np.testing.assert_allclose(freq, two_cards, atol=5 * np.sqrt(0.25 / n))
    np.testing.assert_allclose(np.bincount(dealers, minlength=6)[2:] / n, 0.25, atol=5 * np.sqrt(0.25 / n))
    assert max(totals) <= 10  # 11 is only reachable by drawing


@pytest.mark.parametrize("visible", [2, 3, 4, 5])
def test_dealer_final_distribution(visible: int) -> None:
    dist = dealer_final_distribution(visible)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert min(dist) >= 9
    assert max(dist) <= 8 + 5  # the dealer draws only below 9
    rng, n = np.random.default_rng(visible), 100_000
    finals = []
    for _ in range(n):
        total = visible + int(rng.integers(2, 6))
        while total < 9:
            total += int(rng.integers(2, 6))
        finals.append(total)
    freq = np.bincount(finals, minlength=14) / n
    for total, p in dist.items():
        assert abs(freq[total] - p) < 5 * np.sqrt(p * (1 - p) / n) + 1e-9


def test_terminal_rules() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        assert ENV.step_one(ENV.encode(11, 4), DRAW, rng) == (ENV.encode(11, 4), 1.0, True)  # always busts
        next_state, cost, done = ENV.step_one(ENV.encode(4, 2), DRAW, rng)  # never busts
        assert not done
        assert cost == 0.0
        assert ENV.decode(next_state)[1] == 2  # the dealer card is kept
        _, cost, done = ENV.step_one(ENV.encode(7, 3), STOP, rng)
        assert done
        assert cost in (-1.0, 0.0, 1.0)


def test_model_rows() -> None:
    model = ENV.model()
    P = model.dense_transitions()
    np.testing.assert_allclose(P[:, STOP].sum(axis=1), 0.0)  # stopping always ends the hand
    for s in range(32):
        total, _ = ENV.decode(s)
        bust = sum(total + card > 11 for card in range(2, 6)) / 4
        assert P[s, DRAW].sum() == pytest.approx(1 - bust)
        assert model.costs[s, DRAW] == pytest.approx(bust)
        assert -1 <= model.costs[s, STOP] <= 1


def test_simulator_matches_model() -> None:
    model, rng, n = ENV.model(), np.random.default_rng(2), 100_000
    P = model.dense_transitions()
    for s in (ENV.encode(8, 3), ENV.encode(10, 5), ENV.encode(5, 2)):
        for a in (STOP, DRAW):
            next_states, costs, done = ENV.step(np.full(n, s), np.full(n, a), rng)
            assert abs(costs.mean() - model.costs[s, a]) < 5 * costs.std() / np.sqrt(n) + 1e-9
            freq = np.bincount(next_states[~done], minlength=32) / n
            assert np.abs(freq - P[s, a]).max() < 5 * np.sqrt(0.25 / n)


def test_known_optimal_value() -> None:
    """Regression value: optimal discounted cost 0.275 at (8, 3); the uniform policy scores 0.540625."""
    scorer = ExactScorer(ENV)
    assert scorer.optimal == pytest.approx(0.275, abs=1e-9)
    assert scorer.score(np.full((32, 2), 0.5)) == pytest.approx(0.540625, abs=1e-9)


@pytest.mark.parametrize(("state", "action"), [(32, 0), (-1, 1), (0, 2)])
def test_invalid_state_or_action(state: int, action: int) -> None:
    with pytest.raises(ValueError, match="Invalid state or action"):
        ENV.step_one(state, action, np.random.default_rng(0))
