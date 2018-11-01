import numpy as np


def func(x):
    return x


def func_der(x):
    return 1


def generate_synthetic(users, projects):
    return [[(1, 1, 1)]]


def generate_vectors(users_num, projects_num, dim):
    return [np.random.normal(0, 0.2, dim) for _ in range(users_num)], \
           [np.random.normal(0, 0.2, dim) for _ in range(projects_num)]


def vecs_dist(vecs1, vecs2):
    sum = 0
    for i in range(len(vecs1)):
        for j in range(len(vecs1[0])):
            sum += abs(vecs1[i][j] - vecs2[i][j])
    return sum


def correct_derivative_test():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    X = generate_synthetic(users, projects)
    inf_users, inf_projects = func(X)
    assert vecs_dist(users, inf_users) < 0.1
    assert vecs_dist(projects, inf_projects) < 0.1
