"""Memory-bounded evaluation of the unchanged finite mixture decision equations."""
import numpy as np
from paper_study.decision_test.common import copied_function
from paper_study.decision_study import inference as original


def branch_values(snapshot, branches, cfg):
    chunk = cfg['posterior_chunk_size']

    def posterior(logprior, predicted, observed, sigma):
        # Keep the original E x K draws intact; only evaluate independent K slices.
        return np.concatenate([original.posterior(logprior, predicted, observed[:, start:start + chunk], sigma)
                               for start in range(0, observed.shape[1], chunk)], axis=1)

    adapted = copied_function(original.branch_values, {'posterior': posterior})
    return adapted(snapshot, branches, cfg)
