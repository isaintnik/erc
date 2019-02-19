import numpy as np
from collections import Counter

from src.test.python.basic_synthetic import interaction_matrix
from src.main.python.data_generator import generate_sythetic
from src.main.python.lambda_strategy import NotLookAheadLambdaStrategy


def equal_interaction_test():
    users_num = 1
    projects_num = 3
    dim = 2
    beta = 0.001
    other_project_importance = 0.1
    lambda_transform = abs
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    user_embedding = {k: np.ones((dim,)) * 0.15 for k in range(users_num)}
    project_embedding = {k: np.ones((dim,)) * 0.05 for k in range(projects_num)}
    data = generate_sythetic(user_embedding, project_embedding, beta=beta,
                             other_project_importance=other_project_importance,
                             lambda_strategy_constructor=lambda_strategy_constructor,
                             lambda_transform=lambda_transform, max_lifetime=1000)
    print("|Events| = {}".format(len(data)))
    print(interaction_matrix(user_embedding, project_embedding))
    print(Counter([session.pid for session in data]))
    print()


def different_interaction_one_sign_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.1
    lambda_transform = abs
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    u_e = np.ones((dim,)) * 0.1
    user_embedding = {0: u_e}
    project_embedding = {0: u_e * 2, 1: u_e, 2: u_e}
    data = generate_sythetic(user_embedding, project_embedding, beta=beta,
                             other_project_importance=other_project_importance,
                             lambda_strategy_constructor=lambda_strategy_constructor,
                             lambda_transform=lambda_transform, max_lifetime=1000)
    print("|Events| = {}".format(len(data)))
    print(interaction_matrix(user_embedding, project_embedding))
    print(Counter([session.pid for session in data]))
    print()


def different_interaction_diff_sign_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.1
    lambda_transform = abs
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    u_e = np.ones((dim,)) * 0.1
    user_embedding = {0: u_e}
    project_embedding = {0: u_e, 1: -u_e, 2: -u_e}
    data = generate_sythetic(user_embedding, project_embedding, beta=beta,
                             other_project_importance=other_project_importance,
                             lambda_strategy_constructor=lambda_strategy_constructor,
                             lambda_transform=lambda_transform, max_lifetime=1000)
    print("|Events| = {}".format(len(data)))
    print(interaction_matrix(user_embedding, project_embedding))
    print(Counter([session.pid for session in data]))
    print()


def short_and_long_interaction_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.1
    lambda_transform = abs
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    u_e1 = np.ones((dim,)) * 0.12
    u_e2 = np.ones((dim,)) * 0.1
    user_embedding = {0: u_e1, 1: u_e2}
    project_embedding = {0: u_e1, 1: u_e2, 2: u_e2 - np.ones((dim,)) * 0.02}
    data = generate_sythetic(user_embedding, project_embedding, beta=beta,
                             other_project_importance=other_project_importance,
                             lambda_strategy_constructor=lambda_strategy_constructor,
                             lambda_transform=lambda_transform, max_lifetime=1000)
    print("|Events| = {}".format(len(data)))
    print(interaction_matrix(user_embedding, project_embedding))
    print(Counter([event.pid for event in data if event.uid == 0]))
    print(Counter([event.pid for event in data if event.uid == 1]))
    print()


if __name__ == "__main__":
    equal_interaction_test()
    different_interaction_one_sign_test()
    different_interaction_diff_sign_test()
    short_and_long_interaction_test()
