
from collections import Counter
from src.main.python.data_generator import *

# TS_START = 0
# TS_END = 1
# PROJECT_ID = 2
# N_DONE = 3


# class TestDataGenerator(unittest.TestCase):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dg = DataGenerator(n_users=30, n_projects=30, n_samples=10000)
#         self.users_history = self.dg.generate_users_history()
#
#     def test_types(self):
#
#         self.assertIsInstance(self.users_history, dict)
#         for user_history in self.users_history.values():
#             self.assertIsInstance(user_history, list)
#             for t in user_history:
#                 self.assertIsInstance(t, tuple)
#                 for i in t:
#                     self.assertIsInstance(i, int)
#
#     def test_n_done(self):
#         for user_history in self.users_history.values():
#
#             for pr, gr in groupby(sorted(user_history, key=lambda step: step[PROJECT_ID]),
#                                   key=lambda step: step[PROJECT_ID]):
#                 step_idx = 0
#                 for step in gr:
#                     self.assertEqual(step_idx, step[N_DONE])
#                     step_idx += 1
#
#     def test_time(self):
#
#         for user_history in self.users_history.values():
#
#             for pr, gr in groupby(sorted(user_history, key=lambda step: (step[PROJECT_ID], step[N_DONE])),
#                                   key=lambda step: step[PROJECT_ID]):
#
#                 prev_step_ts_start, prev_step_ts_end = 0, 1
#                 for step in gr:
#
#                     step_ts_start, step_ts_end = step[TS_START], step[TS_END]
#
#                     self.assertLessEqual(step_ts_start, step_ts_end)
#                     self.assertLessEqual(prev_step_ts_end , step_ts_start)
#
#                     prev_step_ts_start, prev_step_ts_end = step_ts_start, step_ts_end
#

def equal_interaction_test():
    dim = 2
    beta = 0.001
    u_e = np.ones((dim,)) * 0.5
    p_e = [np.ones((dim,)) * 0.2] * 3
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e, beta=beta, other_project_importance=0.3,
                       max_lifetime=500000, verbose=False)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg.generate_user_steps()
    print(Counter([session.pid for session in gen_summary]))
    print()


def different_interaction_one_sign_test():
    dim = 2
    beta = 0.001
    u_e = np.ones((dim,)) * 0.4
    p_e = [u_e * 2, u_e, u_e]
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e, beta=beta, other_project_importance=0.3,
                       max_lifetime=500000)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg.generate_user_steps()
    print(Counter([session.pid for session in gen_summary]))
    print()


def different_interaction_diff_sign_test():
    dim = 2
    beta = 0.001
    u_e = np.ones((dim,)) * 0.2
    p_e = [u_e, -u_e, -u_e]
    sg = StepGenerator(user_embedding=u_e, project_embeddings=p_e, beta=beta, other_project_importance=0,
                       max_lifetime=500000, verbose=False)
    print([project_embedding @ sg.user_embedding for project_embedding in sg.project_embeddings])
    gen_summary = sg.generate_user_steps()
    print(Counter([session.pid for session in gen_summary]))
    print()


def short_and_long_interaction_test():
    dim = 2
    beta = 0.001
    u_e1 = np.ones((dim,)) * 0.7
    u_e2 = np.ones((dim,)) * 0.6
    u_e = [u_e1, u_e2]
    p_e = [u_e1, u_e2, u_e2 - np.ones((dim,)) * 0.1]
    X = [StepGenerator(user_embedding=vec, project_embeddings=p_e, beta=beta, other_project_importance=0.3,
                       max_lifetime=500000).generate_user_steps() for vec in u_e]
    print([project_embedding @ u_e1 for project_embedding in p_e])
    print(Counter([session.pid for session in X[0]]))
    print([project_embedding @ u_e2 for project_embedding in p_e])
    print(Counter([session.pid for session in X[1]]))
    print()


if __name__ == "__main__":
    equal_interaction_test()
    different_interaction_one_sign_test()
    different_interaction_diff_sign_test()
    short_and_long_interaction_test()
