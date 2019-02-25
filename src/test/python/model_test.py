import numpy as np

from src.main.python.data_generator import generate_sythetic
from src.main.python.lambda_strategy import NotLookAheadLambdaStrategy
from src.main.python.model import ApplicableModel
from src.main.python import user_lambda


EPS = 1e-9


def test_lambda_growth():
    beta = 0.2
    other_project_importance = 0.1
    lambda_transform = abs
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    user_embedding = {0: np.array([0.1, 0.2])}
    project_embedding = {0: np.array([0.2, 0.0])}
    data = generate_sythetic(user_embedding, project_embedding, beta=beta,
                             other_project_importance=other_project_importance, lambda_transform=lambda_transform,
                             lambda_strategy_constructor=lambda_strategy_constructor, max_lifetime=10000)
    print("data len", len(data))
    model = ApplicableModel(user_embedding, project_embedding, beta=beta,
                            other_project_importance=other_project_importance, lambda_transform=lambda_transform,
                            lambda_strategy_constructor=lambda_strategy_constructor)
    start_lambda = model.get_lambda(0, 0)
    model.fit(data)
    end_lambda = model.get_lambda(0, 0)
    print(end_lambda, start_lambda)
    assert end_lambda > start_lambda


def test_lambda():
    beta = .2
    other_project_importance = .1
    users = {1: np.array([1, 0.5])}
    projects = {
        1: np.array([1, 1]),
        2: np.array([-1, -1])
    }
    interactions_calculator = user_lambda.SimpleInteractionsCalculator(users, projects)
    lambda_calculator = user_lambda.UserLambda(users[1], projects, beta,
                                               other_project_importance, interactions_calculator.get_user_supplier(1))
    lambda_calculator.update(1)
    assert abs(lambda_calculator.get_lambda(1) -
               (interactions_calculator.get_interaction(1, 1) + interactions_calculator.get_interaction(1, 1))) < EPS
    assert abs(lambda_calculator.get_lambda(2) -
               (interactions_calculator.get_interaction(1, 1) * other_project_importance +
                interactions_calculator.get_interaction(1, 2))) < EPS

    lambda_calculator.update(2)
    assert abs(lambda_calculator.get_lambda(1) - (
          np.exp(-beta) * interactions_calculator.get_interaction(1, 1) * other_project_importance +
          interactions_calculator.get_interaction(1, 1) * (1 - other_project_importance) +
          other_project_importance * interactions_calculator.get_interaction(1, 2) +
          interactions_calculator.get_interaction(1, 1)
    )) < EPS
    assert abs(lambda_calculator.get_lambda(2) - (
            np.exp(-beta) * interactions_calculator.get_interaction(1, 1) * other_project_importance +
            other_project_importance * interactions_calculator.get_interaction(1, 2) +
            interactions_calculator.get_interaction(1, 2) * (1 - other_project_importance) +
            interactions_calculator.get_interaction(1, 2)
    )) < EPS


def test_lambda_decrease():
    beta = .2
    other_project_importance = .1
    users = {1: np.array([1, 0.5])}
    projects = {
        1: np.array([1, 1]),
        2: np.array([-1, -1])
    }
    interactions_calculator = user_lambda.SimpleInteractionsCalculator(users, projects)
    lambda_calculator = user_lambda.UserLambda(users[1], projects, beta,
                                               other_project_importance, interactions_calculator.get_user_supplier(1))
    lambda_calculator.update(1)
    lambda1 = lambda_calculator.get_lambda(1)
    lambda_calculator.update(2)
    lambda2 = lambda_calculator.get_lambda(1)
    assert lambda2 < lambda1


if __name__ == "__main__":
    test_lambda_growth()
    test_lambda()
    test_lambda_decrease()
