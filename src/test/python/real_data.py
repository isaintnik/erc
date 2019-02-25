import pickle
import random
import time
import argparse

import numpy as np

from src.main.python.data_preprocess.toloka import toloka_read_raw_data, toloka_prepare_data
from src.main.python.data_preprocess.lastfm import lastfm_read_raw_data, lastfm_prepare_data
from src.main.python.data_preprocess.common import filter_data, train_test_split
from src.main.python.lambda_strategy import NotLookAheadLambdaStrategy
from src.main.python.metrics import return_time_mae, item_recommendation_mae, unseen_recommendation
from src.main.python.model import Model, reduplicate

TOLOKA_FILENAME = "~/data/mlimlab/erc/datasets/toloka/toloka_2018_10_01_2018_11_01_salt_simple_merge"
LASTFM_FILENAME = "~/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_1M.tsv"
# LASTFM_FILENAME = 'data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname_1M.tsv'


def train(model, data, eval, learning_rate, iter_num, optimization_type="sgd", model_path_in=None, model_path_out=None):
    loaded = False
    if model is None and model_path_in is not None:
        try:
            with open(model_path_in, 'rb') as model_file:
                model = pickle.load(model_file)
                print('Model loaded')
                loaded = True
        except FileNotFoundError:
            print('Model file not found')
    if model is None and not loaded:
        raise AttributeError("model is not provided")

    print("start ll = {}, return_time = {}, recommendation_mae = {}".format(
        model.log_likelihood(data),
        return_time_mae(model.get_applicable(data), eval),
        item_recommendation_mae(model.get_applicable(data), eval)
    ))

    for top in [1, 5]:
        unseen_rec = unseen_recommendation(model.get_applicable(data), data, eval, top=top)
        print("unseen_recs@{}: {}".format(str(top), unseen_rec))
    # for top in [1, 5]:
    #     unseen_rec = unseen_recommendation_random(model.get_applicable(data), data, eval, top=top)
    #     print("unseen_recs_random@{}: {}".format(str(top), unseen_rec))

    model.fit(data, learning_rate, iter_num, optimization_type, eval, verbose=True)

    if model_path_out is not None:
        try:
            with open(model_path_out, 'wb') as model_file:
                pickle.dump(model, model_file)
        except FileNotFoundError:
            print('Model saving failed')

    return model


def toloka_test():
    dim = 10
    beta = 0.001
    other_project_importance = 0.1
    eps = 1
    lambda_transform = lambda x: x ** 2
    lambda_derivative = lambda x: 2 * x
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    size = 1000 * 1000
    train_ratio = 0.75
    users_num = 1000
    projects_num = 3000
    top_items = True
    model_path_in = None
    model_path_out = None  # "saved_models/toloka_rand.model"

    raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
    X = toloka_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, top=top_items, users_num=users_num, projects_num=projects_num)
    print("Events after filter", len(X))

    X_tr, X_te = train_test_split(X, train_ratio)
    model = Model(dim, eps=eps, beta=beta, other_project_importance=other_project_importance,
                  lambda_transform=lambda_transform, lambda_derivative=lambda_derivative,
                  lambda_strategy_constructor=lambda_strategy_constructor)
    learning_rate = 3
    print("Params: dim={}, size={}, users_num={}, projects_num={}, lr={}"
          .format(dim, size, users_num, projects_num, learning_rate))
    train(model, X_tr, X_te, learning_rate, iter_num=5, optimization_type="sgd", model_path_in=model_path_in,
          model_path_out=model_path_out)

    # learning_rate = 0.001
    # model = train(None, X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num=2,
    #               optimization_type="glove", model_path_in=model_path_in, model_path_out=model_path_out)

    # print_metrics(model, X_te, X_tr=X_tr, samples_num=samples_num)


def make_data(data_path, size, users_num, projects_num, top_items, train_ratio):
    raw_data = lastfm_read_raw_data(data_path, size)
    X = lastfm_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, top=top_items, users_num=users_num, projects_num=projects_num)
    return train_test_split(X, train_ratio)


def lastfm_test(data_path, model_load_path, model_save_path, iter_num):
    dim = 10
    beta = 0.1
    other_project_importance = 0.1
    eps = 1
    lambda_strategy_constructor = NotLookAheadLambdaStrategy
    size = 100 * 1000
    train_ratio = 0.75
    users_num = 1000
    projects_num = 1000
    optimization_type = "sgd"
    # iter_num = 50
    learning_rate = 1e-4
    top_items = True
    lambda_transform = np.square
    lambda_derivative = reduplicate

    X_tr, X_te = make_data(data_path, size, users_num, projects_num, top_items, train_ratio)

    if model_load_path is not None:
        try:
            with open(model_load_path, 'rb') as model_file:
                model = pickle.load(model_file)
                print('Model loaded')
        except FileNotFoundError:
            print('Model file not found')
            return
    else:
        model = Model(dim=dim, eps=eps, beta=beta, other_project_importance=other_project_importance,
                      lambda_transform=lambda_transform, lambda_derivative=lambda_derivative,
                      lambda_strategy_constructor=lambda_strategy_constructor)

    print("Params: dim={}, size={}, users_num={}, projects_num={}, lr={}"
          .format(dim, size, users_num, projects_num, learning_rate))
    train(model, X_tr, X_te, learning_rate, iter_num=iter_num, optimization_type=optimization_type,
          model_path_in=model_load_path, model_path_out=model_save_path)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--data', default=LASTFM_FILENAME)
    argument_parser.add_argument('--model_load', default=None)
    argument_parser.add_argument('--model_save', default=None)
    argument_parser.add_argument('--iterations', default=50, type=int)
    args = argument_parser.parse_args()

    np.random.seed(3)
    random.seed(3)
    start_time = time.time()
    #toloka_test()
    lastfm_test(args.data, args.model_load, args.model_save, args.iterations)
    print("time:", time.time() - start_time)
