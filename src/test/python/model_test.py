import numpy as np

from src.main.python.data_generator import generate_sythetic
from src.main.python.lambda_strategy import NotLookAheadLambdaStrategy
from src.main.python.model import ApplicableModel


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


if __name__ == "__main__":
    test_lambda_growth()