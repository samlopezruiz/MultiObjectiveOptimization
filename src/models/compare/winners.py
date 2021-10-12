import numpy as np
import pandas as pd

from src.utils.stats import wilcoxon_rank


class Winners:
    def __init__(self, metrics, labels):
        '''
        metrics: rows are problem samples, cols are algorithms
        '''
        self.cd_mat_calculated = False
        self.labels = labels
        self.rankings = np.empty_like(metrics).astype(int)
        self.metrics = metrics
        self.n_elements = metrics.shape[1]
        self.ranking_count = np.zeros((self.n_elements, self.n_elements))
        self.condorcet_mat = np.zeros((self.n_elements, self.n_elements))
        self.calc_rankings()
        self.count_rankings()

    def calc_rankings(self):
        for i, res in enumerate(self.metrics):
            self.rankings[i, :] = np.argsort(np.argsort(res))

    def count_rankings(self):
        for a in range(self.n_elements):
            for r in range(self.n_elements):
                self.ranking_count[r, a] = np.sum(self.rankings[:, a] == r)

    def get_ranking_count(self):
        pos = np.array(range(self.n_elements)) + 1
        return pd.DataFrame(self.ranking_count,
                            columns=self.labels,
                            index=['{} place'.format(p) for p in pos]).astype(int)

    def border_count(self):
        weights = np.array(range(self.n_elements))[::-1] + 1
        border_score = np.multiply(self.ranking_count, weights.reshape(-1, 1))
        ix = np.argsort(np.sum(border_score, axis=0))
        bd = pd.DataFrame(np.sum(border_score, axis=0), columns=['score'], index=self.labels)

        return [self.labels[i] for i in ix[::-1]], bd

    def condorcet_matrix(self, problem_samples=None, alternative='less'):
        '''
        :param problem_samples: shape [problems, algos, n_samples]
        :param alternative: what is better, 'less' or 'greater'
        :return: condorcet_matrix
        '''
        if problem_samples is None:
            for x in range(self.n_elements):
                for y in range(self.n_elements):
                    for res in self.rankings:
                        self.condorcet_mat[x, y] += int(res[x] < res[y])
        else:
            wr_mat = np.zeros((problem_samples.shape[0], self.n_elements, self.n_elements))
            for i, algos_hvs in enumerate(problem_samples):
                wr_mat[i, :, :] = wilcoxon_rank(algos_hvs, self.labels, alternative=alternative).values
            p_value = 0.05
            wr_mat_test = (wr_mat <= p_value).astype(int)
            self.condorcet_mat = np.sum(wr_mat_test, axis=0)

        self.cd_mat_calculated = True

    def get_condorcet(self, problem_samples=None, alternative='less'):
        if not self.cd_mat_calculated:
            self.condorcet_matrix(problem_samples, alternative)
        return pd.DataFrame(self.condorcet_mat,
                            columns=self.labels,
                            index=self.labels).astype(int)

    def condercet_winner(self, problem_samples=None, alternative='less'):
        if not self.cd_mat_calculated:
            self.condorcet_matrix(problem_samples, alternative)

        ind_wins = np.zeros((self.n_elements, self.n_elements))
        for x in range(self.n_elements):
            for y in range(self.n_elements):
                if x != y:
                    ind_wins[x, y] += int(self.condorcet_mat[x, y] >
                                          self.condorcet_mat[y, x])
        ind_wins = np.sum(ind_wins, axis=1)
        cs = pd.DataFrame(ind_wins, columns=['individual wins'], index=self.labels).astype(int)
        condorcet_winner = [self.labels[i] for i, wins in enumerate(ind_wins) if wins == self.n_elements - 1]
        return condorcet_winner, cs

    def score(self, problem_samples=None, alternative='less'):
        sorted_algos, border_scores = self.border_count()
        condorcet_winner, pairwise_wins = self.condercet_winner(problem_samples, alternative)
        return {
            'ranking_counts': self.get_ranking_count(),
            'sorted_algos': sorted_algos,
            'border_scores': border_scores,
            'condorcet_scores': self.get_condorcet(),
            'condorcet_winner': condorcet_winner,
            'pairwise_wins': pairwise_wins
        }

