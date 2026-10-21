import unittest
from training.evaluation.ppo_fresh_counterfactual import classify


class FreshCounterfactualTests(unittest.TestCase):
    def test_observed_survival_harm_and_censoring(self):
        base = {'pieces': 512, 'scores': list(range(512)), 'fours': [0]*512,
                'game_over': False, 'censored': True}
        other = {'pieces': 200, 'scores': list(range(200)), 'fours': [0]*200,
                 'game_over': True, 'censored': False}
        result = classify(base, other)
        self.assertTrue(result['survival_harm'])
        self.assertTrue(result['ppo_censored'])
        other['game_over'] = False
        self.assertFalse(classify(base, other)['survival_harm'])

    def test_common_window_four_loss(self):
        base = {'pieces': 600, 'scores': [10]*600, 'fours': [min(i//100, 5) for i in range(600)],
                'game_over': False, 'censored': True}
        other = {'pieces': 600, 'scores': [10]*600, 'fours': [min(i//200, 2) for i in range(600)],
                 'game_over': False, 'censored': True}
        result = classify(base, other)
        self.assertTrue(result['four_harm'])
        self.assertEqual(result['common_pieces'], 600)


if __name__ == '__main__': unittest.main()
