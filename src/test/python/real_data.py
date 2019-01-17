import time
import math
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from src.main.python.model import USession, Model2Lambda


# filenames
TOLOKA_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/sessions_2018_10_01_2018_10_02"
LASTFM_FILENAME = "/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/" \
                  "userid-timestamp-artid-artname-traid-traname_1M.tsv"


# toloka

def toloka_read_raw_data(filename, size=None):
    raw_data = pd.read_json(filename, lines=True).values
    return raw_data if size is None else raw_data[:size]


def toloka_raw_to_session(raw, project_to_index):
    project_id = raw[3]
    start_ts = raw[4] / (60 * 60)
    end_ts = raw[0] / (60 * 60)
    pr_delta = raw[1] / (60 * 60)
    n_tasks = raw[2]
    user_id = raw[5]
    if project_id not in project_to_index:
        project_to_index[project_id] = len(project_to_index)
    return user_id, USession(project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


def toloka_prepare_data(data):
    users_history = {}
    user_to_index = {}
    project_to_index = {}
    pr_deltas = []
    for val in data:
        user_id, session = toloka_raw_to_session(val, project_to_index)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history[user_to_index[user_id]] = []
        users_history[user_to_index[user_id]].append(session)
        if session.pr_delta is not None and not math.isnan(session.pr_delta):
            pr_deltas.append(session.pr_delta)
    pr_deltas = np.array(pr_deltas)
    print("mean pr_delta", np.mean(pr_deltas), np.std(pr_deltas))
    return users_history


# last.fm

# first 100k raws
# raw_data = pd.read_csv(LASTFM_FILENAME, sep='\t', error_bad_lines=False)
# raw_data.head(100000).to_csv("/Users/akhvorov/data/mlimlab/erc/datasets/lastfm-dataset-1K/"
#                              "userid-timestamp-artid-artname-traid-traname_100000.tsv", sep='\t')

def lastfm_read_raw_data(filename, size=None):
    raw_data = pd.read_csv(filename, sep='\t').values
    return raw_data if size is None else raw_data[:size]


def lastfm_raw_to_session(raw, project_to_index, last_time_done):
    ts = raw[2]
    project_id = raw[3]
    if project_id not in project_to_index:
        project_to_index[project_id] = len(project_to_index)
    start_ts = ts / (60 * 60)
    end_ts = 0  # don't used
    pr_delta = None if project_to_index[project_id] not in last_time_done \
        else (ts - last_time_done[project_to_index[project_id]]) / (60 * 60)
    n_tasks = 1
    user_id = raw[1]
    last_time_done[project_to_index[project_id]] = ts
    return user_id, USession(project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


def lastfm_make_sessions(users_history):
    new_history = {}
    for user_id, user_history in users_history.items():
        new_history[user_id] = []
        prev_session = None
        for i, session in enumerate(user_history):
            if prev_session is None or prev_session.pid == session.pid:
                pass


def lastfm_prepare_data(data):
    data[:, 2] = np.array(list(map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")), data[:, 2])))
    data = data[np.argsort(data[:, 2])]
    print("Max time delta =", np.max(data[:, 2]) - np.min(data[:, 2]))
    users_history = {}
    user_to_index = {}
    project_to_index = {}
    last_time_done = {}
    # make combination to session
    last_session = None
    pr_deltas = []
    for val in data:
        user_id, session = lastfm_raw_to_session(val, project_to_index, last_time_done)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history[user_to_index[user_id]] = []
        if last_session is not None and last_session.pid == session.pid:
            continue
        users_history[user_to_index[user_id]].append(session)
        last_session = session
        if session.pr_delta is not None:
            pr_deltas.append(session.pr_delta)
    # print(project_to_index)
    pr_deltas = np.array(pr_deltas)
    print("Mean pr_delta = {}, std = {}".format(np.mean(pr_deltas), np.std(pr_deltas)))
    return users_history


# common

def interaction_matrix(users, projects):
    users = np.array(list(users.values()))
    projects = np.array(list(projects.values()))
    return users @ projects.T


def top_data(data):
    users_stat = {}
    projects_stat = {}
    for uid, user_history in data.items():
        users_stat[uid] = len(user_history)
        for session in user_history:
            projects_stat.setdefault(session.pid, 0)
            projects_stat[session.pid] += 1
    users_stat = list(users_stat.items())
    projects_stat = list(projects_stat.items())
    sorted(users_stat, key=lambda x: -x[1])
    sorted(projects_stat, key=lambda x: -x[1])
    return [uid for (uid, value) in users_stat], [pid for (pid, value) in projects_stat]


def random_data(data):
    users_stat = set()
    projects_stat = set()
    for uid, user_history in data.items():
        users_stat.add(uid)
        for session in user_history:
            projects_stat.add(session.pid)
    users_stat = list(users_stat)
    projects_stat = list(projects_stat)
    random.shuffle(users_stat)
    random.shuffle(projects_stat)
    return users_stat, projects_stat


def filter_data(data, users_num=None, projects_num=None):
    users_stat, projects_stat = top_data(data)
    user_num = len(users_stat) if users_num is None else min(users_num, len(users_stat))
    projects_num = len(projects_stat) if projects_num is None else min(projects_num, len(projects_stat))
    selected_users = set(users_stat[:user_num])
    selected_projects = set(projects_stat[:projects_num])
    before_events_num, after_events_num = 0, 0
    for uid in list(data.keys()):
        before_events_num += len(data[uid])
        if uid in selected_users:
            user_history = data[uid]
            data[uid] = []
            for session in user_history:
                if session.pid in selected_projects:
                    data[uid].append(session)
                    after_events_num += 1
        else:
            del data[uid]
    print("Events num before = {}, after = {}".format(before_events_num, after_events_num))
    return data


def get_split_time(data, train_ratio):
    start_times = []
    for user_id, user_history in data.items():
        for session in user_history:
            start_times.append(session.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = {}, {}
    split_time = get_split_time(data, train_ratio)
    # print("split time", split_time)
    for user_id, user_history in data.items():
        if user_history[0].start_ts >= split_time or user_history[-1].start_ts <= split_time:
            continue
        train[user_id] = []
        test[user_id] = []
        for i, session in enumerate(user_history):
            if session.start_ts < split_time:
                train[user_id].append(session)
            else:
                test[user_id].append(session)
    return train, test


def sgd_optimization(model, data, eval, iter_num):
    for i in range(iter_num):
        # if i % 5 == 0 or i in [1, 2]:
        #     print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model.optimization_step()
        print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model_application = model.get_applicable()
        return_time = return_time_mae(model_application, eval, samples_num=10)
        print("return_time:", return_time)
        print()
    # print(interaction_matrix(model.user_embeddings, model.project_embeddings))
    # print(np.mean(interaction_matrix(model.user_embeddings, model.project_embeddings)))


def train(data, eval, dim, beta, other_project_importance, learning_rate, iter_num, optimization_type="sgd",
          model_filename=None, load=False):
    loaded = False
    if load:
        try:
            with open(model_filename, 'rb') as model_file:
                model = pickle.load(model_file)
                print('Model loaded')
                loaded = True
        except TypeError:
            print('Model filename not passed')
        except FileNotFoundError:
            print('Model not found, a new one was created')
    if not loaded:
        model = Model2Lambda(data, dim, learning_rate=learning_rate, eps=1, beta=beta,
                             other_project_importance=other_project_importance)

    print("start ll = {}, return_time = {}".format(model.log_likelihood(),
                                                   return_time_mae(model.get_applicable(), eval, samples_num=10)))
    if optimization_type == "glove":
        model.glove_like_optimisation(iter_num=iter_num, verbose=True, eval=eval)
    elif optimization_type == "sgd":
        sgd_optimization(model, data, eval, iter_num)
    elif "mix":
        k = 2
        model.glove_like_optimisation(iter_num=k, verbose=True, eval=eval)
        model.learning_rate = 0.01
        sgd_optimization(model, data, eval, iter_num - k)

    try:
        with open(model_filename, 'wb') as model_file:
            pickle.dump(model, model_file)
    except TypeError:
        print('Filename for model saving is not passed')
    except FileNotFoundError:
        print('Model saving failed')

    model_application = model.get_applicable()
    return model_application


def return_time_mae(model, data, samples_num=10):
    errors = 0.
    count = 0
    for user_id, user_history in data.items():
        for i, session in enumerate(user_history):
            if i > 0 and session.pr_delta is not None and not math.isnan(session.pr_delta) and session.n_tasks > 0:
                count += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                # expected_return_time = 0
                errors += abs(expected_return_time - session.pr_delta)
            model.accept(user_id, session)
    return errors / count


def where_fails(model, data, samples_num=10):
    session_predictions = []
    zero_pr_delta = 0
    split_time = get_split_time(data, 0.7)
    for user_id, user_history in data.items():
        for i, session in enumerate(user_history):
            if session.start_ts < split_time:
                continue
            if i > 0 and session.pr_delta is not None and not math.isnan(session.pr_delta) and session.n_tasks > 0:
                if session.pr_delta < 1e-5:
                    zero_pr_delta += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                # expected_return_time = 0
                session_predictions.append((expected_return_time, session, i))
            model.accept(user_id, session)
    # session_predictions = sorted(session_predictions, key=lambda x: -abs(x[0] - x[1].pr_delta))
    print("zero pr_delta rate:", zero_pr_delta / len(session_predictions), zero_pr_delta, len(session_predictions))
    for pr_time, session, ind in session_predictions:
        print(session, pr_time, ind)


def real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio,
                   optimization_type, model_filename=None, load=False):
    X_tr, X_te = train_test_split(X, train_ratio)
    model_application = train(X_tr, X_te, dim, beta, other_project_importance, learning_rate, iter_num,
                              optimization_type, model_filename, load)
    return_time = return_time_mae(model_application, X_te, samples_num=samples_num)
    print("return_time:", return_time)
    # where_fails(model_application, X, samples_num=samples_num)


def toloka_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.2
    # learning_rate = 1.5
    learning_rate = 0.1
    optimization_type = "sgd"
    iter_num = 2
    size = 10000
    samples_num = 10
    train_ratio = 0.7
    raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
    X = toloka_prepare_data(raw_data)
    print("objects_num:", size)
    print("users_num:", len(X))
    real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio,
                   optimization_type)


def lastfm_test():
    dim = 15
    beta = 0.001
    other_project_importance = 0.1
    learning_rate = 0.0003
    optimization_type = "mix"
    iter_num = 10
    size = 1 * 1000 * 1000
    samples_num = 10
    train_ratio = 0.75
    users_num = 1000
    projects_num = 3000
    model_filename = "lastfm_1M_1k_3k_top_2.model"
    load = False
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    X = lastfm_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X = filter_data(X, users_num=users_num, projects_num=projects_num)
    print("Users num:", len(X))
    real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio,
                   optimization_type, model_filename, load=load)


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
        train(x_train, x_test, dim=5, beta=0.001, other_project_importance=0.1, learning_rate=0.001, iter_num=20,
              optimization_type='glove', model_filename=arguments.model_path)
    else:
        raise NotImplementedError()


def main_test(arguments):
    if arguments.data_type == 'lastfm':
        lastfm_test()
    else:
        toloka_test()


if __name__ == "__main__":
    np.random.seed(3)
    random.seed(3)
    argument_parser = argparse.ArgumentParser()
    subparsers = argument_parser.add_subparsers()

    eval_parser = subparsers.add_parser('eval')
    eval_parser.set_defaults(func=main_eval)

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=main_train)

    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=main_test)

    argument_parser.add_argument('data_type')
    argument_parser.add_argument('data_path')
    argument_parser.add_argument('model_path')

    args = argument_parser.parse_args()

    start_time = time.time()
    # toloka_test()
    # lastfm_test()
    args.func(args)
    print("time:", time.time() - start_time)
