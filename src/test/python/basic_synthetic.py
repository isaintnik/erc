import numpy as np
from src.main.python.model import Model, USession


def generate_synthetic(users, projects):
    return [
        [USession(0, 10, 15, None, 10), USession(1, 16, 18, None, 5),
         USession(2, 25, 30, None, 10)],
        [USession(2, 10, 16, None, 8), USession(0, 20, 28, None, 16),
         USession(2, 36, 40, 10, 8), USession(2, 48, 60, 50, 24)]
    ]


def generate_vectors(users_num, projects_num, dim, std_dev=0.2):
    return [np.random.normal(0, std_dev, dim) for _ in range(users_num)], \
           [np.random.normal(0, std_dev, dim) for _ in range(projects_num)]


def vecs_dist(vecs1, vecs2):
    sum = 0
    for i in range(len(vecs1)):
        sum += abs(vecs1[i] - vecs2[i])
    return sum


def correct_test():
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


def convergence_test():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic(users, projects)
    model = Model(users_history, dim, learning_rate=0.01)
    for i in range(100):
        print("{}-th optimization step".format(i))
        print("users:", vecs_dist(model.get_user_embeddings(), users))
        print("projects:", vecs_dist(model.get_project_embeddings(), projects))
        model.optimization_step()

    assert vecs_dist(model.get_user_embeddings(), users) < 0.1
    assert vecs_dist(model.get_project_embeddings(), projects) < 0.1


if __name__ == "__main__":
    correct_test()
