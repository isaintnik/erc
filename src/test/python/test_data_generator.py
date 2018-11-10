import unittest
from itertools import groupby

from src.main.python.data_generator import DataGenerator

TS_START = 0
TS_END = 1
PROJECT_ID = 2
N_DONE = 3


class TestDataGenerator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dg = DataGenerator(n_users=30, n_projects=30, n_samples=10000)
        self.users_history = self.dg.generate_users_history()

    def test_types(self):

        self.assertIsInstance(self.users_history, dict)
        for user_history in self.users_history.values():
            self.assertIsInstance(user_history, list)
            for t in user_history:
                self.assertIsInstance(t, tuple)
                for i in t:
                    self.assertIsInstance(i, int)

    def test_n_done(self):
        for user_history in self.users_history.values():

            for pr, gr in groupby(sorted(user_history, key=lambda step: step[PROJECT_ID]),
                                  key=lambda step: step[PROJECT_ID]):
                step_idx = 0
                for step in gr:
                    self.assertEqual(step_idx, step[N_DONE])
                    step_idx += 1

    def test_time(self):

        for user_history in self.users_history.values():

            for pr, gr in groupby(sorted(user_history, key=lambda step: (step[PROJECT_ID], step[N_DONE])),
                                  key=lambda step: step[PROJECT_ID]):

                prev_step_ts_start, prev_step_ts_end = 0, 1
                for step in gr:

                    step_ts_start, step_ts_end = step[TS_START], step[TS_END]

                    self.assertLessEqual(step_ts_start, step_ts_end)
                    self.assertLessEqual(prev_step_ts_end , step_ts_start)

                    prev_step_ts_start, prev_step_ts_end = step_ts_start, step_ts_end