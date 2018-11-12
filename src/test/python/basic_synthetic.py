import numpy as np
from src.main.python.model import Model, USession


def generate_synthetic(users, projects):
    return [
        [USession(0, 10, 15, None, 10), USession(1, 16, 18, None, 5),
         USession(2, 25, 30, None, 10), USession(1, 36, 38, 20, 3), USession(1, 44, 49, 6, 7)],
        [USession(2, 10, 16, None, 8), USession(0, 20, 28, None, 16),
         USession(2, 36, 40, 10, 8), USession(2, 48, 60, 50, 24)]
    ]


def generate_vectors(users_num, projects_num, dim, std_dev=0.2):
    return np.array([np.random.normal(0, std_dev, dim) for _ in range(users_num)]), \
           np.array([np.random.normal(0, std_dev, dim) for _ in range(projects_num)])


def vecs_dist(vecs1, vecs2):
    sum = 0
    for i in range(len(vecs1)):
        sum += abs(vecs1[i] - vecs2[i])
    return sum


def correct_likelihood_test():
    m = Model(generate_synthetic(None, None), 2, eps=10)
    log_likelihood = m.log_likelihood()
    print(log_likelihood)
    user_derivatives, project_derivatives = m.ll_derivative()
    print(user_derivatives)
    print(project_derivatives)
    m.user_embeddings += np.ones_like(m.user_embeddings) * 0.0001
    second_log_likelihood = m.log_likelihood()
    print(second_log_likelihood)
    print(second_log_likelihood - log_likelihood)


def correct_derivative_test():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic(users, projects)
    model = Model(users_history, dim, learning_rate=0.01, eps=10)
    user_embeddings1 = model.get_user_embeddings()
    print("truth")
    print(users)
    print()
    print("init emb")
    print(user_embeddings1)
    # project_embeddings1 = model.get_project_embeddings()
    # print(project_embeddings1)
    users_derivatives, project_derivatives = model.ll_derivative()
    print()
    print("derivative")
    print(users_derivatives)
    # print(project_derivatives)
    users_delta = 0.001 * (users - user_embeddings1)
    print()
    print("delta")
    print(users_delta)
    model.user_embeddings += users_delta
    users_derivatives, project_derivatives = model.ll_derivative()
    # projects_delta = 0.001 * (projects - project_embeddings1)
    # print(projects_delta)
    # for (user_derivative, user_delta) in zip(users_derivatives, users_delta):
    #     assert np.abs(user_derivative - user_delta) < 0.0001
    # for (project_derivative, project_delta) in zip(project_derivatives, projects_delta):
    #     assert np.abs(project_derivative - project_delta) < 0.0001


def convergence_test():
    users_num = 2
    projects_num = 2
    dim = 3
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic(users, projects)
    model = Model(users_history, dim, learning_rate=0.003, eps=10)
    for i in range(100):
        if i % 5 == 0:
            print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model.optimization_step()
    print(model.user_embeddings)
    print(model.project_embeddings)


if __name__ == "__main__":
    # correct_derivative_test()
    convergence_test()
