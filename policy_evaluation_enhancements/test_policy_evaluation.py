"""Independent checks for the lecture-aligned experiments."""
import unittest
import numpy as np
from .policy_evaluation import (GridWorld, Blackjack, reference_model, exact_policy_value,
    optimal_value, stationary_weights, feature_matrix, td_step, run_control,
    evaluate_fixed, fixed_training_data, tomc_start, tomc_observe)
from .utils import truncated_first_visit_q


class EvaluationTests(unittest.TestCase):
    def test_scalar_sampler_matches_model(self):
        rng = np.random.default_rng(29)
        for env, states in [(GridWorld(), [0, 9, 12, 55, 99]), (Blackjack(), [0, 13, 28, 31])]:
            P, c = reference_model(env)
            for s in states:
                for a in range(env.n_actions):
                    counts = np.zeros(env.n_states)
                    costs = 0.
                    for _ in range(16000):
                        ns, cost, done = env.step_one(s, a, rng)
                        if not done:
                            counts[ns] += 1
                        costs += cost
                    np.testing.assert_allclose(counts/16000, P[:,s,a], atol=.018)
                    self.assertAlmostEqual(costs/16000, c[s,a], delta=.035)

    def test_diagnostic_references(self):
        for env in [GridWorld(), Blackjack()]:
            P,c = reference_model(env)
            pi = np.full(c.shape, 1/env.n_actions)
            V,Q = exact_policy_value(P,c,pi,env.gamma)
            np.testing.assert_allclose(V,(pi*Q).sum(axis=1),atol=1e-11)
            Vstar = optimal_value(P,c,env.gamma)
            bellman = np.min(c+env.gamma*np.einsum('nsa,n->sa',P,Vstar),axis=1)
            np.testing.assert_allclose(Vstar,bellman,atol=1e-11)
            self.assertTrue(np.all(V >= Vstar-1e-10))
            w = stationary_weights(env,P,pi)
            self.assertAlmostEqual(w.sum(),1.)
            self.assertTrue(np.all(w>=0))

    def test_true_online_forward_view_and_td0(self):
        rng=np.random.default_rng(99)
        xs=rng.normal(size=(6,4));xs/=np.linalg.norm(xs,axis=1,keepdims=True);xs[-1]=0
        costs=rng.normal(size=5);initial=rng.normal(size=4)
        alpha,gamma,lam=.13,.9,.7
        theta=initial.copy();e=np.zeros(4);old=0.;history=[initial.copy()]
        for h in range(1,6):
            old=td_step(theta,e,old,xs[h-1],xs[h],costs[h-1],gamma,alpha,lam,True)
            forward=initial.copy()
            for t in range(h):
                target=0.
                for n in range(1,h-t+1):
                    ret=sum(gamma**j*costs[t+j] for j in range(n))
                    ret+=gamma**n*float(history[t+n-1]@xs[t+n])
                    coeff=lam**(n-1) if n==h-t else (1-lam)*lam**(n-1)
                    target+=coeff*ret
                forward+=alpha*(target-float(forward@xs[t]))*xs[t]
            np.testing.assert_allclose(theta,forward,atol=1e-12)
            history.append(theta.copy())
        a=initial.copy();b=initial.copy();ea=np.zeros(4);eb=np.zeros(4);oa=ob=0.
        for t in range(5):
            oa=td_step(a,ea,oa,xs[t],xs[t+1],costs[t],gamma,alpha,0.,False)
            ob=td_step(b,eb,ob,xs[t],xs[t+1],costs[t],gamma,alpha,0.,True)
            np.testing.assert_allclose(a,b,atol=1e-12)

    def test_streaming_tomc_terminal_and_discount(self):
        pi=np.ones((2,1));data=tomc_start(pi,.5)
        for s,cost,done in [(0,.2,False),(1,.4,True),(0,.2,False),(1,.4,True)]:
            tomc_observe(data,s,0,cost,done,3)
        self.assertTrue(data['complete'])
        q=truncated_first_visit_q(data,pi,.5,3,data['threshold'])
        np.testing.assert_allclose(q[:,0],[.4,.4])

    def test_independent_budgets_and_reproducibility(self):
        env=GridWorld();P,c=reference_model(env);vs=optimal_value(P,c,.9)
        kwargs=dict(budget=80,checkpoints=[40,80],seed=3,step_size=.1,eval_steps=20,
                    P=P,c=c,V_star=vs,T=40,score_rollouts=2)
        mc=run_control(env,'tomc',0,0,**kwargs)
        td=run_control(env,'td_tabular',.1,.5,**kwargs)
        self.assertEqual(mc[-1]['observations'],80)
        self.assertEqual(td[-1]['observations'],80)
        self.assertEqual(mc[-1]['policy_updates'],0)
        self.assertEqual(td[-1]['policy_updates'],4)
        self.assertEqual(td,run_control(env,'td_tabular',.1,.5,**kwargs))
        pi=np.full((100,4),.25);_,q=exact_policy_value(P,c,pi,.9);w=stationary_weights(env,P,pi)
        paths=fixed_training_data(env,80,3)
        a,ha=evaluate_fixed(env,'td_linear',.1,.5,80,[80],3,q,w,paths=paths)
        b,hb=evaluate_fixed(env,'td_linear',.1,.5,80,[80],3,q,w)
        np.testing.assert_allclose(a,b)
        self.assertEqual(ha,hb)


if __name__=='__main__':
    unittest.main()
