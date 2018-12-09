import numpy as np
from src.main.python import lambda_calc
from src.test.python.basic_synthetic import generate_vectors, generate_history


EPS = 1e-6


def test_one_project_py_cpp():
    users_num = 5
    projects_num = 3
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    users, projects = generate_vectors(users_num, projects_num, dim, std_dev=0.2)
    history = generate_history(users=users, projects=projects, beta=beta,
                               other_project_importance=other_project_importance)
    interactions = lambda_calc.InteractionCalculator(users, projects)
    lambdas = lambda_calc.calc_lambdas(0, 0, history[0], users[0], beta, interactions, projects, dim)

    project_index = lambda_calc.projects_index(history[0])
    rev_projects_index = lambda_calc.reverse_projects_indices(project_index)
    project_ids, time_deltas, n_tasks = lambda_calc.convert_history(history[0], rev_projects_index)
    native_lambdas = lambda_calc.calc_lambdas_native(rev_projects_index[0], project_ids, n_tasks, time_deltas, users[0],
                                                     dim, beta, interactions.interactions[0][project_index],
                                                     projects[project_index])
    delta = np.abs(lambdas - native_lambdas)
    assert delta.max() < EPS


def test_users_derivative():
    users_num = 5
    projects_num = 3
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    users, projects = generate_vectors(users_num, projects_num, dim, std_dev=0.2)
    history = generate_history(users=users, projects=projects, beta=beta,
                               other_project_importance=other_project_importance)
    interactions = lambda_calc.InteractionCalculator(users, projects)
    _, user_derivatives, _ = lambda_calc.calc_lambdas(0, 0, history[0], users[0], beta, interactions, projects, dim,
                                                      derivative=True)

    project_index = lambda_calc.projects_index(history[0])
    rev_projects_index = lambda_calc.reverse_projects_indices(project_index)
    project_ids, time_deltas, n_tasks = lambda_calc.convert_history(history[0], rev_projects_index)
    _, native_user_derivatives, _ = \
        lambda_calc.calc_lambdas_native(rev_projects_index[0], project_ids, n_tasks, time_deltas, users[0], dim, beta,
                                        interactions.interactions[0][project_index], projects[project_index],
                                        derivative=True)
    delta = np.abs(user_derivatives - native_user_derivatives)
    print(delta.mean(), delta.max(), delta.min())
    assert delta.max() < EPS


if __name__ == '__main__':
    test_one_project_py_cpp()
    test_users_derivative()
