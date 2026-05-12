# A/B Testing Framework (Phase 8)

import numpy as np
import scipy.stats as stats

from .evaluate import ndcg_at_k


class ABTestFramework:
    """Simulated offline A/B test comparing two recommendation models.

    Computes per-user metric scores for both control and treatment,
    then runs a paired t-test (same users see both models, so paired
    removes user-level variance and is more statistically powerful).
    """

    def __init__(self, metric_fn=None, k: int = 10):
        """
        metric_fn : callable(recommended_ids, relevant_set) -> float
                    Defaults to NDCG@k.
        k         : rank cutoff used by the default metric.
        """
        self.k         = k
        self.metric_fn = metric_fn or (lambda recs, rel: ndcg_at_k(recs, rel, k))

    def run(self, control_fn, treatment_fn, val_df, n_users: int = 500) -> dict:
        """Run the A/B test and return a result dict.

        control_fn   : callable(user_id) -> [movieId, ...]
        treatment_fn : callable(user_id) -> [movieId, ...]
        val_df       : DataFrame with userId and movieId columns (ground truth)
        n_users      : how many val users to evaluate
        """
        val_items = val_df.groupby('userId')['movieId'].apply(set).to_dict()
        users     = list(val_items.keys())[:n_users]

        control_scores, treatment_scores = [], []
        for uid in users:
            relevant = val_items.get(uid, set())
            if not relevant:
                continue
            control_scores.append(self.metric_fn(control_fn(uid), relevant))
            treatment_scores.append(self.metric_fn(treatment_fn(uid), relevant))

        return self._analyze(control_scores, treatment_scores)

    def _analyze(self, control: list, treatment: list) -> dict:
        ctrl  = np.array(control)
        treat = np.array(treatment)

        ctrl_mean  = float(np.mean(ctrl))
        treat_mean = float(np.mean(treat))
        lift_pct   = (treat_mean - ctrl_mean) / (ctrl_mean + 1e-10) * 100

        # Paired t-test on per-user differences
        t_stat, p_value = stats.ttest_rel(ctrl, treat)

        # 95% confidence interval on the mean difference
        diff = treat - ctrl
        ci   = stats.t.interval(0.95, df=len(diff) - 1,
                                 loc=float(np.mean(diff)),
                                 scale=stats.sem(diff))

        return {
            'n_users':        len(ctrl),
            'control_mean':   round(ctrl_mean, 4),
            'treatment_mean': round(treat_mean, 4),
            'lift_pct':       round(lift_pct, 2),
            'p_value':        round(float(p_value), 4),
            'significant':    bool(p_value < 0.05),
            'ci_95':          (round(float(ci[0]), 4), round(float(ci[1]), 4)),
        }
