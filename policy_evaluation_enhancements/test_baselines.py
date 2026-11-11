"""Run with: python3 -m unittest test_baselines.py"""
import unittest
from functools import lru_cache
import numpy as np
from policy_evaluation_enhancements.baseline import GridWorld
from policy_evaluation_enhancements.baseline_blackjack import Blackjack
from policy_evaluation_enhancements.utils import (monte_carlo_q, mirror_step, sample_returns, train,
                   estimate_Q_online_mc_dynamic, collect_on_policy_steps)


class CycleEnvironment:
    n_states = 3
    n_actions = 1
    gamma = .5

    def initial_states(self, count, rng):
        return np.zeros(count, dtype=int)

    def step(self, states, actions, rng):
        costs = np.array([.2, .4, .6])[states]
        return (states+1) % self.n_states, costs, np.zeros(len(states), dtype=bool)


class TwoStepHands(CycleEnvironment):
    n_states = 2

    def deal(self, count, rng):
        return np.zeros(count, dtype=int)

    def step(self, states, actions, rng):
        return 1-states, np.array([.2, .4])[states], states == 1


class BaselineTests(unittest.TestCase):
    def test_grid_reset_and_costs(self):
        env = GridWorld()
        rng = np.random.default_rng(7)
        for state, cost in [(9, 0), (12, 1)]:
            nxt, costs, done = env.step(np.full(50000, state), np.zeros(50000, dtype=int), rng)
            self.assertAlmostEqual(float(np.mean(nxt == 0)), .8, delta=.01)
            self.assertTrue(np.all(costs == cost))
            self.assertFalse(done.any())
        nxt, _, _ = GridWorld(intended_probability=1).step(np.array([0]), np.array([0]), rng)
        self.assertEqual(nxt[0], 0)

    def test_blackjack_draw_bust_and_end(self):
        env = Blackjack()
        rng = np.random.default_rng(8)
        state = int(env.encode(11, 3))
        _, costs, done = env.step(np.full(100, state), np.ones(100, dtype=int), rng)
        self.assertTrue(done.all())
        self.assertTrue(np.all(costs == 1))
        policy = np.tile([0., 1.], (32, 1))
        returns, count = sample_returns(env, policy, np.full(100, state), 8, rng)
        self.assertEqual(count, 100)  # No moves sampled after terminal.
        self.assertTrue(np.all(returns == 1))

    def test_blackjack_against_exact_enumeration(self):
        env = Blackjack()
        policy = np.full((32, 2), .5)

        @lru_cache(None)
        def dealer_cost(player, dealer):
            if dealer >= 9:
                return -1. if dealer > 11 else float(np.sign(dealer-player))
            return sum(dealer_cost(player, dealer+card) for card in range(2, 6))/4

        @lru_cache(None)
        def q_exact(player, visible, action):
            if action == 0:
                return sum(dealer_cost(player, visible+card) for card in range(2, 6))/4
            return sum(1. if player+card > 11 else .9*sum(
                q_exact(player+card, visible, a) for a in range(2))/2 for card in range(2, 6))/4

        q, count = monte_carlo_q(env, policy, 20000, 8, np.random.default_rng(9))
        exact = np.array([[q_exact(*map(int, env.decode(s)), a) for a in range(2)] for s in range(32)])
        self.assertLess(float(np.max(abs(q-exact))), .025)
        self.assertLess(count, 32*2*20000*8)
        updated = mirror_step(policy, exact, .5)
        self.assertTrue(np.allclose(updated.sum(axis=1), 1))
        self.assertTrue(np.all(updated >= 0))
        self.assertTrue(np.all(np.sum(updated*exact, axis=1) <= np.sum(policy*exact, axis=1)+1e-12))

    def test_budget_and_reproducibility(self):
        env = Blackjack()
        args = (env, 3000, 2, 4, 50, 8, .5, 13)
        p, hist = train(*args)
        p2, hist2 = train(*args)
        np.testing.assert_array_equal(p, p2)
        self.assertEqual(hist, hist2)
        self.assertLessEqual(hist[-1]['cumulative_training_observations'], 3000)
        self.assertEqual(sum(r['training_observations'] for r in hist), hist[-1]['cumulative_training_observations'])
        self.assertEqual(sum(r['training_observations']+r['evaluation_observations'] for r in hist), hist[-1]['cumulative_observations'])


    def test_dynamic_coverage_and_relative_discount(self):
        env = CycleEnvironment()
        pi = np.ones((3, 1))
        q, state, info = estimate_Q_online_mc_dynamic(env, pi, 3, np.random.default_rng(0), 20,
                                                     return_paths=True)
        self.assertTrue(info['complete'])
        self.assertEqual(info['observations'], 5)
        self.assertEqual(info['exploration_observations'], 3)
        self.assertEqual(info['tail_observations'], 2)
        self.assertEqual(len(info['paths']), 5)
        np.testing.assert_allclose(q[:, 0], [.55, .75, .8])
        self.assertEqual(state, 2)

    def test_dynamic_failure_counts_and_no_update(self):
        env = CycleEnvironment()
        for budget, status in [(2, 'incomplete_coverage'), (4, 'incomplete_return_tail')]:
            q, _, info = estimate_Q_online_mc_dynamic(env, np.ones((3, 1)), 3,
                                                      np.random.default_rng(0), budget)
            self.assertIsNone(q)
            self.assertEqual(info['observations'], budget)
            self.assertEqual(info['status'], status)
        pi, history = train(env, 4, 1, 3, 2, 3, .5, 1)
        self.assertEqual(history[-1]['iteration'], 0)
        self.assertEqual(history[-1]['cumulative_training_observations'], 4)
        self.assertEqual(history[-1]['status'], 'incomplete_return_tail')

    def test_dynamic_terminal_and_continuation(self):
        env = TwoStepHands()
        q, _, info = estimate_Q_online_mc_dynamic(env, np.ones((2, 1)), 4,
                                                  np.random.default_rng(0), 20, coverage_rounds=2)
        np.testing.assert_allclose(q[:, 0], [.4, .4])
        self.assertEqual(info['completed_rounds'], 2)
        self.assertEqual(info['observations'], 10)
        paths, _ = collect_on_policy_steps(env, np.ones((2, 1)), 6, np.random.default_rng(0))
        self.assertEqual([p[-1] for p in paths], [True, False, True, False, True, False])

    def test_low_probability_action_bound(self):
        env = CycleEnvironment()
        env.n_actions = 2
        pi = np.tile([1., 0.], (3, 1))
        q, _, info = estimate_Q_online_mc_dynamic(env, pi, 3, np.random.default_rng(0), 20)
        self.assertEqual(info['required_pairs'], 3)
        np.testing.assert_allclose(q[:, 1], 2.)


if __name__ == '__main__':
    unittest.main()
