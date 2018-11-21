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

    def _select_project(self, projects_ids, last_time_done, current_ts):

        # def next_time(project_id):
        #     first_delta = np.random.exponential(scale=1 / self.user_lambdas.get(project_id) ** 2)
        #     delta = first_delta
        #     while last_time_done[pid] + delta < current_ts:
        #         delta = np.random.exponential(scale=1 / self.user_lambdas.get(project_id) ** 2)
        #     return delta, first_delta
        #
        # nearest_pid = (projects_ids[0], 1e9, 1e9)
        # for pid in projects_ids:
        #     pr_delta, inner_delta = next_time(pid)
        #     if last_time_done[pid] + pr_delta < nearest_pid[1]:
        #         nearest_pid = (pid, pr_delta, inner_delta)
        # return nearest_pid

        # for pid in projects_ids:
        #     if 1 / self.user_lambdas.get(pid) ** 2 < 0:
        #         print(1 / self.user_lambdas.get(pid) ** 2)
        #         assert 1 / self.user_lambdas.get(pid) ** 2
        time_deltas = {pid: np.random.exponential(scale=1 / self.user_lambdas.get(pid) ** 2) for pid in projects_ids}
        # time_deltas = {pid: np.random.exponential(scale=1 / np.exp(self.user_lambdas.get(pid))) for pid in projects_ids}
        pid_chosen, time_delta = min(time_deltas.items(), key=lambda item: item[1])
        if self.verbose:
            print({pid: 1 / self.user_lambdas.get(pid) ** 2 for pid in projects_ids})
            # print({pid: 1 / np.exp(self.user_lambdas.get(pid)) for pid in projects_ids})
            print({pid: self.user_lambdas.get(pid) for pid in projects_ids})
        return pid_chosen, time_delta, time_delta

    def generate_user_steps(self):
        current_ts = 0
        generation_summary = []
        projects_ids = list(range(len(self.project_embeddings)))
        last_time_done = {pid: 0 for pid in projects_ids}
        latest_done_project_ts = 0

        k = 0
        while current_ts < self.max_lifetime and k < 100:
            # k += 1

            pid, time_delta, inner_delta = self._select_project(projects_ids, last_time_done, current_ts)
            if self.verbose:
                print("1.", pid, time_delta, inner_delta)
            n_tasks = np.random.geometric(1 / AVG_TASKS_ON_SESSION)
            # n_tasks = 1
            session_time = sum([abs(np.random.normal(AVG_TIME_FOR_TASK, STD_TIME_FOR_TASK)) + 2 for _ in range(n_tasks)])
            # session_time = AVG_TIME_FOR_TASK
            if self.verbose:
                print("2.", n_tasks, session_time)

            # ts_start = last_time_done[pid] + time_delta
            ts_start = current_ts + time_delta
            ts_end = ts_start + session_time
            last_time_done[pid] += time_delta + session_time
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

        return generation_summary
