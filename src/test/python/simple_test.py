import argparse

import numpy as np

from src.main.python.model import Model, Event
from src.main.python.lambda_strategy import NotLookAheadLambdaStrategy
from src.main.python.metrics import return_time_mae
from src.test.python.basic_synthetic import generate_vectors


def little_synthetic():
    users_num = 1
    projects_num = 2
    dim = 5
    beta = 0.001
    eps = 1
    other_project_importance = 0.
    embedding_mean = 0.01
    std_dev = 0.005
    lambda_transform = lambda x: x ** 2
    lambda_derivative = lambda x: 2 * x
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    learning_rate = 1e-4
    inner_iter_num = 1000
    outer_iter_num = 50

    user_init, projects_init = generate_vectors(users_num, projects_num, dim, mean=embedding_mean, std_dev=std_dev)
    history = [
        Event(0, 0, 3600, None, 1),
        Event(0, 0, 7200, 3600, 1),
    ]

    model = Model(dim, beta, eps, other_project_importance, lambda_transform=lambda_transform,
                  lambda_derivative=lambda_derivative, lambda_strategy_constructor=lambda_strategy_constructor,
                  users_embeddings_prior=user_init, projects_embeddings_prior=projects_init)
    print(return_time_mae(model.get_applicable(history), history))
    for i in range(outer_iter_num):
        model.fit(history, learning_rate, iter_num=inner_iter_num, verbose=False)
        applicable = model.get_applicable(history)
        print(model.log_likelihood(history), applicable.time_delta(0, 0), return_time_mae(applicable, history))


def little_real():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    synthetic_parser = subparsers.add_parser('synth')
    synthetic_parser.set_defaults(func=little_synthetic)

    real_parser = subparsers.add_parser('real')
    real_parser.set_defaults(func=real_parser)

    args = parser.parse_args()
    args.func()
