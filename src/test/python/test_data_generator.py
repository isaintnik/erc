from collections import Counter
from src.main.python.data_generator import *


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
                       max_lifetime=500000, verbose=False)
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
