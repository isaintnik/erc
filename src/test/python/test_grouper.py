import os
import random
import string
import unittest
from itertools import groupby

from src.main.python.user_steps_grouper import UserStepSample, log_grouper, UserStep

USER_ID = "user_id"
TS_START = 'ts_start'
TS_END = 'ts_end'
PROJECT_ID = 'project_id'
fields_lst = [USER_ID, TS_START, TS_END, PROJECT_ID]


class TestLogGrouper(unittest.TestCase):

    @classmethod
    def setUp(cls):

        cur_dir = os.getcwd()
        log_path = cur_dir + "/log.txt"
        TestLogGrouper.generate_raw_samples(log_path, n_users=3, n_projects=3, n_samples=10000)
        samples = []
        with open(log_path, 'r') as f:
            for line in f:
                samples.append(UserStepSample(line.strip()))

            cls.steps = log_grouper(samples)

    def test_types(self):

        self.assertIsInstance(self.steps, dict)
        for val in self.steps.values():
            self.assertIsInstance(val, list)
            for item in val:
                self.assertIsInstance(item, UserStep)

    def test_n_done(self):
        for steps in self.steps.values():

            for pr, gr in groupby(sorted(steps, key=lambda st: st.project_id), key=lambda st: st.project_id):
                step_idx = 0
                for step in gr:
                    self.assertEqual(step_idx, step.n_done)
                    step_idx += 1

    def test_time(self):

        for steps in self.steps.values():

            for pr, gr in groupby(sorted(steps, key=lambda st: st.project_id), key=lambda st: st.project_id):
                _ts = next(gr).ts_start
                for step in gr:
                    cur_ts = step.ts_start
                    self.assertLess(_ts, cur_ts)
                    _ts = cur_ts

    @staticmethod
    def generate_raw_samples(log_path, n_users=10, n_projects=10, n_samples=100):
        user_ids = set()
        while len(user_ids) < n_users:
            user_ids.add(TestLogGrouper._generate_user_id())
        user_ids = list(user_ids)

        ts = random.randint(0, 10000)

        with open(log_path, 'w') as f:
            for _ in range(n_samples):
                ts += random.randint(0, 10000)
                ts_start = ts
                ts += random.randint(0, 100)
                ts_end = ts

                project_id = random.randint(0, n_projects - 1)
                user_id = random.choice(user_ids)

                field_vals = [user_id, ts_start, ts_end, project_id]
                s = [field_str + "=" + str(field_val) for field_str, field_val in zip(fields_lst, field_vals)]

                f.write("\t".join(s) + "\n")

    @staticmethod
    def _generate_user_id(id_len=8):
        return "".join(random.choice(string.ascii_letters) for _ in range(
            id_len))
