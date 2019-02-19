import pickle
import random
import time

import numpy as np

from src.main.python.model import Model, ModelExpLambda
from src.main.python.data_preprocess.toloka import toloka_read_raw_data, toloka_prepare_data
from src.main.python.data_preprocess.lastfm import lastfm_read_raw_data, lastfm_prepare_data
from src.main.python.data_preprocess.common import filter_data, train_test_split
from src.main.python.metrics import return_time_mae, item_recommendation_mae, unseen_recommendation

TOLOKA_FILENAME = "~/data/mlimlab/erc/datasets/toloka/toloka_2018_10_01_2018_11_01_salt_simple_merge"
LASTFM_FILENAME = "~/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_1M.tsv"
# LASTFM_FILENAME = 'data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname_1M.tsv'


def train(model, data, eval, learning_rate, iter_num, optimization_type="sgd", model_path_in=None, model_path_out=None,
          samples_num=10):
    loaded = False
    if model is None and model_path_in is not None:
        try:
            with open(model_path_in, 'rb') as model_file:
                model = pickle.load(model_file)
                print('Model loaded')
                loaded = True
        except FileNotFoundError:
            print('Model not found, a new one was created')
    if model is None and not loaded:
        raise AttributeError("model is not provided")

    print("start ll = {}, return_time = {}, recommendation_mae = {}".format(
        model.log_likelihood(data),
        return_time_mae(model.get_applicable(data), eval, samples_num),
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
    size = 1000 * 1000
    samples_num = 10
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
    model = ModelExpLambda(dim, eps, beta, other_project_importance)
    learning_rate = 3
    print("Params: dim={}, size={}, users_num={}, projects_num={}, lr={}"
          .format(dim, size, users_num, projects_num, learning_rate))
    train(model, X_tr, X_te, learning_rate, iter_num=5, optimization_type="sgd", samples_num=samples_num,
          model_path_in=model_path_in, model_path_out=model_path_out)

    # learning_rate = 0.001
    # model = train(None, X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num=2,
    #               optimization_type="glove", model_path_in=model_path_in, model_path_out=model_path_out)

    # print_metrics(model, X_te, X_tr=X_tr, samples_num=samples_num)


def lastfm_test():
    dim = 20
    beta = 0.001
    other_project_importance = 0.1
    eps = 1
    size = 100 * 1000
    samples_num = 10
    train_ratio = 0.75
    users_num = 1000
    projects_num = 1000
    iter_num = 100
    learning_rate = 0.01
    top_items = False
    model_path_in = None
    model_path_out = None  # "saved_models/lastfm_1M_1k_3k_top_2.model"
    lambda_transform = abs
    lambda_derivative = np.sign

    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    X = lastfm_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, top=top_items, users_num=users_num, projects_num=projects_num)
    print("Users num:", len(X))

    X_tr, X_te = train_test_split(X, train_ratio)
    model = Model(dim, eps, beta, other_project_importance, lambda_transform=lambda_transform, lambda_derivative=lambda_derivative)

    print("Params: dim={}, size={}, users_num={}, projects_num={}, lr={}"
          .format(dim, size, users_num, projects_num, learning_rate))
    train(model, X_tr, X_te, learning_rate, iter_num=iter_num, optimization_type="glove", samples_num=samples_num,
          model_path_in=model_path_in, model_path_out=model_path_out)

    # learning_rate = 0.001
    # model = train(model, X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num=2,
    #               optimization_type="glove", model_path_in=model_path_in, model_path_out=model_path_out)


if __name__ == "__main__":
    np.random.seed(3)
    random.seed(3)
    start_time = time.time()
    #toloka_test()
    lastfm_test()
    print("time:", time.time() - start_time)
