import math
import random
import string
import time
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import expon

from src.main.python.model import UserLambda, USession

USER_IDX = 0
TS_START_IDX = 1
TS_END_IDX = 2
PROJECT_ID_IDX = 3

TIME_FOR_ONE_TASK = 120


class StepGenerator:

    def __init__(self, user_embedding=None, project_embeddings=None, n_projects=3, dim=10, beta=0.1,
                 other_project_importance=0.8, max_lifetime = 100):

        self.n_projects = n_projects
        self.max_lifetime = max_lifetime
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding if user_embedding is not None else np.random.randn(dim)
        self.project_embeddings = project_embeddings if project_embeddings is not None else [np.random.randn(dim) for _
                                                                                         in range(n_projects)]

    def _generate_user_steps(self):

        self.user_lambdas = UserLambda(user_embedding=self.user_embedding,
                                       project_embeddings=self.project_embeddings,
                                       beta=self.beta,
                                       other_project_importance=self.other_project_importance,
                                       square=False)

        current_ts = 0

        generation_summary = defaultdict(list)
        lifetime = 0
        p_cnt = 0

        while lifetime < self.max_lifetime:

            lambdas = {pid: self.user_lambdas.get(pid, False) for pid in range(len(self.project_embeddings))}

            time_deltas = {pid: expon.rvs(loc=0, scale=1/lmbd**2) for pid, lmbd in lambdas.items()}
            pid_chosen, time_delta = min(time_deltas.items(), key=lambda item: item[1])

            ts_start = current_ts + time_delta
            ts_end = ts_start + TIME_FOR_ONE_TASK

            _time_delta = time_delta if p_cnt > 0 else 0

            usession = USession(pid_chosen,
                                ts_start,
                                ts_end,
                                _time_delta,
                                1)
            p_cnt += 1
            current_ts = ts_end

            self.user_lambdas.update(usession, _time_delta, False)

            lifetime += _time_delta

            generation_summary["pid_chosen"].append(pid_chosen)
            generation_summary["time_delta"].append(time_delta)
            generation_summary["u_history"].append(usession)

        return generation_summary


if __name__ == "__main__":
    u_e = np.random.randn(2)
    p_e = [np.random.randn(2)]*3
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg._generate_user_steps()
    print(Counter(gen_summary['pid_chosen']))
    print()

    u_e = np.random.randn(2)
    p_e = [u_e*2, u_e, u_e]
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e, other_project_importance=0)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg._generate_user_steps()
    print(Counter(gen_summary['pid_chosen']))
    print()

    u_e = np.random.randn(2)
    p_e = [u_e, -u_e, -u_e]
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e, other_project_importance=0)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg._generate_user_steps()
    print(Counter(gen_summary['pid_chosen']))
    print()

