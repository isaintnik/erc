import math
import random
import string
import time
from collections import defaultdict, Counter

USER_IDX = 0
TS_START_IDX = 1
TS_END_IDX = 2
PROJECT_ID_IDX = 3


class DataGenerator:

    def __init__(self, n_users=10, n_projects=10, n_samples=100):

        self.n_users = n_users
        self.n_projects = n_projects
        self.n_samples = n_samples

    def generate_users_history(self):

        steps = self._generate_users_steps()
        users_history = self._group_steps_by_user(steps)

        return users_history

    def _generate_users_steps(self):

        user_ids = DataGenerator._generate_user_ids(self.n_users)
        current_ts = int(time.time())

        steps = []
        for _ in range(self.n_samples):
            ts_start = current_ts + random.randint(0, 100)
            ts_end = ts_start + random.randint(0, 100)

            current_ts = ts_end + random.randint(0, 100)

            project_id = random.randint(0, self.n_projects - 1)
            user_id = random.choice(user_ids)

            step = (user_id, ts_start, ts_end, project_id)

            steps.append(step)

        return steps

    def _group_steps_by_user(self, steps):

        user_history = defaultdict(list)

        user_pr_cnt = Counter()

        for step in steps:
            user_id = step[USER_IDX]
            pr_id = step[PROJECT_ID_IDX]

            ts_start = step[TS_START_IDX]
            ts_end = step[TS_END_IDX]

            key = "|".join(map(str, [user_id, pr_id]))
            n_done = user_pr_cnt[key]
            user_pr_cnt[key] += 1

            user_history[user_id].append((ts_start, ts_end, pr_id, n_done))

        return user_history

    @staticmethod
    def _generate_user_id(id_len=8):
        return "".join(random.choice(string.ascii_letters) for _ in range(
            id_len))

    @staticmethod
    def _generate_user_ids(n_users):
        user_ids = set()
        user_name_len = int(math.log(10000 * n_users, len(string.ascii_letters)))

        while len(user_ids) < n_users:
            user_ids.add(DataGenerator._generate_user_id(user_name_len))
        user_ids = list(user_ids)

        return user_ids

