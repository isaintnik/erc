import argparse
import pickle
import random
import time

import numpy as np

from src.main.python.model import Model, Model2Lambda, ModelExpLambda, ModelDensity
from src.main.python.data_preprocess.toloka import toloka_read_raw_data, toloka_prepare_data
from src.main.python.data_preprocess.lastfm import lastfm_read_raw_data, lastfm_prepare_data
from src.main.python.data_preprocess.common import filter_data, train_test_split
from src.test.python.metrics import return_time_mae, item_recommendation_mae, unseen_recommendation, \
    unseen_recommendation_random

TOLOKA_FILENAME = "~/data/mlimlab/erc/datasets/toloka_2018_10_01_2018_11_01_salt_simple_merge"
LASTFM_FILENAME = "~/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_1M.tsv"


def sgd_optimization(model, eval, iter_num, train_data=None):
    for i in range(iter_num):
        model.optimization_step()
        print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        print_metrics(model, eval, X_tr=train_data, samples_num=10)
        print()
    # print(interaction_matrix(model.user_embeddings, model.project_embeddings))
    # print(np.mean(interaction_matrix(model.user_embeddings, model.project_embeddings)))


def train(model, data, eval, dim, beta, other_project_importance, learning_rate, iter_num, optimization_type="sgd",
          model_path_in=None, model_path_out=None):
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
        model = ModelDensity(data, dim, learning_rate=learning_rate, eps=1, beta=beta,
                             other_project_importance=other_project_importance)

    print("start ll = {}, return_time = {}, recommendation_mae = {}".format(
        model.log_likelihood(),
        return_time_mae(model.get_applicable(), eval, samples_num=10),
        item_recommendation_mae(model.get_applicable(), eval)
    ))

    for top in [1, 5]:
        unseen_rec = unseen_recommendation(model.get_applicable(), data, eval, top=top)
        print("unseen_recs@{}: {}".format(str(top), unseen_rec))
    # for top in [1, 5]:
    #     unseen_rec = unseen_recommendation_random(model.get_applicable(), data, eval, top=top)
    #     print("unseen_recs_random@{}: {}".format(str(top), unseen_rec))

    if optimization_type == "glove":
        model.glove_like_optimisation(iter_num=iter_num, verbose=True, eval=eval)
    elif optimization_type == "sgd":
        sgd_optimization(model, eval, iter_num, train_data=data)

    if model_path_out is not None:
        try:
            with open(model_path_out, 'wb') as model_file:
                pickle.dump(model, model_file)
        except FileNotFoundError:
            print('Model saving failed')

    return model


def print_metrics(model, X_te, X_tr=None, samples_num=10):
    return_time = return_time_mae(model.get_applicable(), X_te, samples_num=samples_num)
    recommend_mae = item_recommendation_mae(model.get_applicable(), X_te)
    if X_tr is not None:
        unseen_rec = unseen_recommendation(model.get_applicable(), X_tr, X_te, top=1)
        unseen_rec_5 = unseen_recommendation(model.get_applicable(), X_tr, X_te, top=5)
        print("return_time = {}, recommendation_mae = {}, unseen_rec = {}, unseen_rec@5 = {}".format(
            return_time, recommend_mae, unseen_rec, unseen_rec_5))
    else:
        print("return_time = {}, recommendation_mae = {}".format(return_time, recommend_mae))


def toloka_test():
    dim = 5
    beta = 0.001
    other_project_importance = 0.1
    size = 1000 * 1000
    samples_num = 10
    train_ratio = 0.75
    users_num = None
    projects_num = None
    top_items = False
    model_path_in = None
    model_path_out = "saved_models/toloka_rand.model"

    raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
    X = toloka_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, top=top_items, users_num=users_num, projects_num=projects_num)

    X_tr, X_te = train_test_split(X, train_ratio)
    model = None

    # learning_rate = 0.001
    # model = train(None, X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num=2,
    #               optimization_type="glove", model_path_in=model_path_in, model_path_out=model_path_out)

    learning_rate = 1.
    model = train(model, X_tr, X_te, dim, beta, other_project_importance, learning_rate=learning_rate, iter_num=5,
                  optimization_type="sgd", model_path_in=model_path_in, model_path_out=model_path_out)

    print_metrics(model, X_te, X_tr=X_tr, samples_num=samples_num)


def lastfm_test():
    dim = 10
    beta = 0.001
    other_project_importance = 0.1
    size = 1 * 100 * 1000
    samples_num = 10
    train_ratio = 0.75
    users_num = 1000
    projects_num = 3000
    top_items = False
    model_path_in = None
    model_path_out = "saved_models/lastfm_1M_1k_3k_top_2.model"

    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    X = lastfm_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, top=top_items, users_num=users_num, projects_num=projects_num)
    print("Users num:", len(X))

    X_tr, X_te = train_test_split(X, train_ratio)
    model = None

    # learning_rate = 0.001
    # model = train(model, X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num=2,
    #               optimization_type="glove", model_path_in=model_path_in, model_path_out=model_path_out)

    learning_rate = 20
    model = train(model, X_tr, X_te, dim, beta, other_project_importance, learning_rate=learning_rate, iter_num=50,
                  optimization_type="sgd", model_path_in=model_path_in, model_path_out=model_path_out)

    print_metrics(model, X_te, X_tr=X_tr, samples_num=samples_num)


def prepare_lastfm_data(filename, size=50000, train_ratio=0.75):
    raw_data = lastfm_read_raw_data(filename, size)
    X = lastfm_prepare_data(raw_data)
    return train_test_split(X, train_ratio)


def main_eval(arguments):
    samples_num = 10
    with open(arguments.model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    if arguments.data_type == 'lastfm':
        _, data = prepare_lastfm_data(arguments.data_path)
    else:
        data = None
    return_time = return_time_mae(model.get_application(), data, samples_num=samples_num)
    print("return_time:", return_time)


def main_train(arguments):
    if arguments.data_type == 'lastfm':
        x_train, x_test = prepare_lastfm_data(arguments.data_path)
        train(x_train, x_test, dim=15, beta=0.001, other_project_importance=0.1, learning_rate=0.001, iter_num=20,
              optimization_type='glove', model_filename=arguments.model_path)
    else:
        raise NotImplementedError()


def main_test(arguments):
    if arguments.data_type == 'lastfm':
        lastfm_test()
    else:
        toloka_test()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--dataset', default="lastfm")
    parser.add_argument('-dp', '--data_path', default="")
    parser.add_argument('-m', '--mode', default="all")
    parser.add_argument('-mi', '--model_path_in', default=None)
    parser.add_argument('-mo', '--model_path_out', default=None)
    parser.add_argument('-wt', '--with_train', action='store_true')
    parser.add_argument('-o', '--optim', default="sgd")
    parser.add_argument('-rt', '--rettime', action='store_true')
    parser.add_argument('-ri', '--recitem', action='store_true')
    parser.add_argument('--dim', default=15)
    parser.add_argument('-i', '--iter', default=10)
    parser.add_argument('-b', '--beta', default=0.001)
    parser.add_argument('-lr', '--learning_rate', default=0.0003)
    return parser


if __name__ == "__main__":
    np.random.seed(3)
    random.seed(3)
    # args = arg_parser().parse_args()
    # argument_parser = argparse.ArgumentParser()
    # subparsers = argument_parser.add_subparsers()
    #
    # eval_parser = subparsers.add_parser('eval')
    # eval_parser.set_defaults(func=main_eval)
    #
    # train_parser = subparsers.add_parser('train')
    # train_parser.set_defaults(func=main_train)
    #
    # test_parser = subparsers.add_parser('test')
    # test_parser.set_defaults(func=main_test)
    #
    # argument_parser.add_argument('data_type')
    # argument_parser.add_argument('data_path')
    # argument_parser.add_argument('model_path')
    #
    # args = argument_parser.parse_args()

    start_time = time.time()
    # toloka_test()
    lastfm_test()
    # args.func(args)
    print("time:", time.time() - start_time)
