import numpy as np
from src.main.python.model import UserLambda, USession

USER_IDX = 0
TS_START_IDX = 1
TS_END_IDX = 2
PROJECT_ID_IDX = 3

AVG_TIME_FOR_TASK = 1
STD_TIME_FOR_TASK = 1
AVG_TASKS_ON_SESSION = 3.73


class StepGenerator:

    def __init__(self, user_embedding=None, project_embeddings=None, n_projects=3, dim=10, beta=0.001,
                 other_project_importance=0.8, max_lifetime=100, verbose=False):
        self.n_projects = n_projects
        self.max_lifetime = max_lifetime
        self.beta = beta
        self.verbose = verbose
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding if user_embedding is not None else np.random.randn(dim)
        self.project_embeddings = project_embeddings if project_embeddings is not None else [np.random.randn(dim) for _
                                                                                             in range(n_projects)]
        self.user_lambdas = UserLambda(user_embedding=self.user_embedding,
                                       n_projects=len(self.project_embeddings),
                                       beta=self.beta,
                                       other_project_importance=self.other_project_importance,
                                       derivative=False,
                                       square=False)

    def _select_project_at_current_time(self, projects_ids):
        time_deltas = {pid: np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2) for pid in projects_ids}
        # time_deltas = {pid: np.random.exponential(scale=1 / np.exp(self.user_lambdas.get(pid))) for pid in projects_ids}
        pid_chosen, time_delta = min(time_deltas.items(), key=lambda item: item[1])
        if self.verbose:
            print({pid: 1 / self.user_lambdas.get(pid) ** 2 for pid in projects_ids})
            # print({pid: 1 / np.exp(self.user_lambdas.get(pid)) for pid in projects_ids})
            print({pid: self.user_lambdas.get(pid) for pid in projects_ids})
        return pid_chosen, time_delta, time_delta

    def _select_project_step_by_step(self, projects_ids, last_time_done, current_ts):
        time_deltas = {pid: last_time_done[pid] + np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2) for pid in projects_ids}
        pid, inner_delta = min(time_deltas.items(), key=lambda item: item[1])
        inner_delta -= last_time_done[pid]
        time_delta = np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2)
        if self.verbose:
            print({pid: 1 / self.user_lambdas.get(pid) ** 2 for pid in projects_ids})
            print({pid: self.user_lambdas.get(pid) for pid in projects_ids})
        return pid, time_delta, inner_delta

    def _select_project_with_future_steps(self, projects_ids, last_time_done, next_time_done, current_ts):
        pid, start_ts = min(next_time_done.items(), key=lambda item: item[1])
        inner_delta = start_ts - last_time_done[pid]
        time_delta = start_ts - current_ts
        last_time_done[pid] = next_time_done[pid]
        next_time_done[pid] = next_time_done[pid] + np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2)
        if self.verbose:
            print({pid: 1 / self.user_lambdas.get(pid) ** 2 for pid in projects_ids})
            print({pid: self.user_lambdas.get(pid) for pid in projects_ids})
        return pid, time_delta, inner_delta

    def generate_user_steps(self):
        current_ts = 0
        generation_summary = []
        projects_ids = list(range(len(self.project_embeddings)))
        last_time_done = {pid: 0 for pid in projects_ids}
        next_time_done = {pid: np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2) for pid in projects_ids}
        latest_done_project_ts = 0

        while current_ts < self.max_lifetime:

            # pid, time_delta, inner_delta = self._select_project_at_current_time(projects_ids)
            pid, time_delta, inner_delta = self._select_project_step_by_step(projects_ids, last_time_done, current_ts)
            # pid, time_delta, inner_delta = self._select_project_with_future_steps(projects_ids, last_time_done,
                                                                                  # next_time_done, current_ts)
            if self.verbose:
                print("1.", pid, time_delta, inner_delta)
            n_tasks = np.random.geometric(1 / AVG_TASKS_ON_SESSION)
            n_tasks = np.log(n_tasks)
            # n_tasks = 1
            session_time = sum([abs(np.random.normal(AVG_TIME_FOR_TASK, STD_TIME_FOR_TASK)) + 1 for _ in range(int(n_tasks))])
            if self.verbose:
                print("2.", n_tasks, session_time)

            # ts_start = last_time_done[pid] + time_delta
            ts_start = current_ts + time_delta
            ts_end = ts_start + session_time
            last_time_done[pid] = ts_end
            current_ts = ts_end

            user_session = USession(pid, ts_start, ts_end, inner_delta, n_tasks)
            if self.verbose:
                print("3.", user_session)
                print(np.exp(-self.beta * (ts_end - latest_done_project_ts)))
            self.user_lambdas.update(self.project_embeddings[pid], user_session, ts_end - latest_done_project_ts)
            latest_done_project_ts = ts_end
            generation_summary.append(user_session)
            if self.verbose:
                print()

        for pid in last_time_done.keys():
            user_session = USession(pid, self.max_lifetime, self.max_lifetime + 1, self.max_lifetime - last_time_done[pid], 0)
            generation_summary.append(user_session)

        print({pid: self.user_lambdas.get(pid) for pid in projects_ids})
        return generation_summary
