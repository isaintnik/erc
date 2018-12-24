import numpy as np
from src.main.python.model import ModelApplication, USession

USER_IDX = 0
TS_START_IDX = 1
TS_END_IDX = 2
PROJECT_ID_IDX = 3

AVG_TIME_FOR_TASK = 1
STD_TIME_FOR_TASK = 1
AVG_TASKS_ON_SESSION = 3.73


class StepGenerator:

    def __init__(self, user_embedding=None, project_embeddings=None, beta=0.001,
                 other_project_importance=0.8, default_lambda=1., start_from=0, max_lifetime=50000, verbose=False):
        self.model = ModelApplication({0: user_embedding}, project_embeddings, beta, other_project_importance,
                                      default_lambda=default_lambda)
        self.n_projects = len(project_embeddings)
        self.start_from = start_from
        self.max_lifetime = max_lifetime
        self.beta = beta
        self.verbose = verbose

    def _select_project_at_current_time(self, projects_ids):
        time_deltas = {pid: self.model.time_delta(0, pid) for pid in projects_ids}
        pid_chosen, time_delta = min(time_deltas.items(), key=lambda item: item[1])
        if self.verbose:
            print({pid: 1 / self.model.get_lambda(0, pid) for pid in projects_ids})
            print({pid: self.model.get_lambda(0, pid) for pid in projects_ids})
        return pid_chosen, time_delta, time_delta

    def _select_project_step_by_step(self, projects_ids, last_time_done, current_ts):
        time_deltas = {pid: last_time_done[pid] + self.model.time_delta(0, pid) for pid in projects_ids}
        pid, inner_delta = min(time_deltas.items(), key=lambda item: item[1])
        inner_delta -= last_time_done[pid]
        time_delta = self.model.time_delta(0, pid)
        if self.verbose:
            print({pid: 1 / self.model.get_lambda(0, pid) for pid in projects_ids})
            print({pid: self.model.get_lambda(0, pid) for pid in projects_ids})
        return pid, time_delta, inner_delta

    def _select_project_with_future_steps(self, projects_ids, last_time_done, next_time_done, current_ts):
        pid, start_ts = min(next_time_done.items(), key=lambda item: item[1])
        inner_delta = start_ts - last_time_done[pid]
        time_delta = start_ts - current_ts
        last_time_done[pid] = next_time_done[pid]
        next_time_done[pid] = next_time_done[pid] + self.model.time_delta(0, pid)
        if self.verbose:
            print({pid: 1 / self.model.get_lambda(0, pid) ** 2 for pid in projects_ids})
            print({pid: self.model.get_lambda(0, pid) for pid in projects_ids})
        return pid, time_delta, inner_delta

    # def _select_project_with_future_steps(self, projects_ids, last_time_done, next_time_done, current_ts):
    #     pid, start_ts = min(next_time_done.items(), key=lambda item: item[1])
    #     inner_delta = start_ts - last_time_done[pid]
    #     time_delta = start_ts - current_ts
    #     last_time_done[pid] = next_time_done[pid]
    #     next_time_done[pid] = next_time_done[pid] + self.model.time_delta(0, pid)
    #     if self.verbose:
    #         print({pid: 1 / self.model.get_lambda(0, pid) ** 2 for pid in projects_ids})
    #         print({pid: self.model.get_lambda(0, pid) for pid in projects_ids})
    #     return pid, time_delta, inner_delta

    def generate_user_steps(self):
        current_ts = 0
        generation_summary = []
        projects_ids = list(range(self.n_projects))
        last_time_done = {pid: 0 for pid in projects_ids}
        next_time_done = {pid: self.model.time_delta(0, pid) for pid in projects_ids}
        latest_done_project_ts = 0

        while current_ts < self.max_lifetime:

            pid, time_delta, inner_delta = self._select_project_at_current_time(projects_ids)
            # pid, time_delta, inner_delta = self._select_project_step_by_step(projects_ids, last_time_done, current_ts)
            # pid, time_delta, inner_delta = self._select_project_with_future_steps(projects_ids, last_time_done,
            #                                                                       next_time_done, current_ts)
            if self.verbose:
                print("1.", pid, time_delta, inner_delta)
            n_tasks = np.random.poisson(AVG_TASKS_ON_SESSION) + 1
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
            self.model.accept(0, user_session)
            latest_done_project_ts = ts_end
            if current_ts > self.start_from:
                generation_summary.append(user_session)
            if self.verbose:
                print()

        for pid in last_time_done.keys():
            user_session = USession(pid, self.max_lifetime, self.max_lifetime + 1, self.max_lifetime - last_time_done[pid], 0)
            generation_summary.append(user_session)

        if self.verbose:
            print({pid: self.model.get_lambda(0, pid) for pid in projects_ids})
        return generation_summary


def generate_history(*, users, projects, beta, other_project_importance, default_lambda, start_from=0, max_lifetime=50000):
    return {user_id: StepGenerator(
        user_embedding=user_embedding, project_embeddings=projects, beta=beta,
        other_project_importance=other_project_importance, default_lambda=default_lambda, start_from=start_from,
        max_lifetime=max_lifetime, verbose=False).generate_user_steps() for user_id, user_embedding in users.items()}
