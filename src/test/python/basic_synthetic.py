import numpy as np
from src.main.python.model import Model


def generate_synthetic(users, projects):
    return [[(0, 0, 0, 0) for _ in range(3)] for _ in range(len(users))]


def generate_vectors(users_num, projects_num, dim, std_dev=0.2):
    return [np.random.normal(0, std_dev, dim) for _ in range(users_num)], \
           [np.random.normal(0, std_dev, dim) for _ in range(projects_num)]


def vecs_dist(vecs1, vecs2):
    sum = 0
    for i in range(len(vecs1)):
        sum += abs(vecs1[i] - vecs2[i])
    return sum


def test_correct_derivative():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic(users, projects)
    model = Model(users_history, dim, learning_rate=0.01)
    user_embeddings1 = model.get_user_embeddings()
    project_embeddings1 = model.get_project_embeddings()
    users_derivatives, project_derivatives = model.calc_derivative()
    users_delta = [0.001 * (user - user_embedding) for user, user_embedding
                   in zip(users, user_embeddings1)]
    projects_delta = [0.001 * (project - project_embedding) for project, project_embedding
                      in zip(projects, project_embeddings1)]
    for (user_derivative, user_delta) in zip(users_derivatives, users_delta):
        assert np.abs(user_derivative - user_delta) < 0.0001
    for (project_derivative, project_delta) in zip(project_derivatives, projects_delta):
        assert np.abs(project_derivative - project_delta) < 0.0001


def test_convergence():
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
