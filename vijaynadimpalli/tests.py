import numpy as np
import pandas as pd
import unittest

from statsmodels.distributions.empirical_distribution import ECDF
from partner_selection import PartnerSelection
from ps_utils import extremal_measure, get_co_variance_matrix, \
    get_sum_correlations_vectorized, diagonal_measure_vectorized, multivariate_rho_vectorized


class PartnerSelectionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.quadruple = ['A', 'AAL', 'AAP', 'AAPL']
        df = pd.read_csv('data/sp500_2016_test.csv', parse_dates=True, index_col='Date').dropna()
        cls.ps = PartnerSelection(df)

        cls.u = cls.ps.returns.apply(lambda x: ECDF(x)(x), axis=0)
        cls.co_variance_matrix = get_co_variance_matrix()

    def test_traditional(self):
        self.assertEqual(self.ps.traditional(1),[['A', 'AAL', 'AAPL', 'AAP']])

    def test_extended(self):
        self.assertEqual(self.ps.extended(1), [['A', 'AAL', 'AAPL', 'AAP']])

    def test_geometric(self):
        self.assertEqual(self.ps.geometric(1),[['A', 'AAL', 'AAPL', 'AAP']])

    def test_extremal(self):
        self.assertEqual(self.ps.extremal(1), [['A', 'AAL', 'AAPL', 'AAP']])

    def test_sum_correlations(self):
        self.assertEqual(round(
            get_sum_correlations_vectorized(self.ps.correlation_matrix.loc[self.quadruple,self.quadruple], np.array([[0,1,2,3]]))[1], 4), 1.9678)

    def test_multivariate_rho(self):
        self.assertEqual(round(multivariate_rho_vectorized(self.u[self.quadruple], np.array([[0,1,2,3]]))[1], 4), 0.3114)

    def test_diagonal_measure(self):
        self.assertEqual(round(diagonal_measure_vectorized(self.ps.ranked_returns[self.quadruple], np.array([[0,1,2,3]]))[1], 4), 91.9374)

    def test_extremal_measure(self):
        self.assertEqual(round(extremal_measure(self.ps.ranked_returns[self.quadruple], self.co_variance_matrix), 4), 108.5128)

if __name__ == '__main__':
    unittest.main()
