import time
import math
import numpy as np
import pandas as pd
from src.main.python.model import USession, Model2Lambda, ModelApplication


# filenames


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
    for val in data:
        user_id, session = toloka_raw_to_session(val, project_to_index)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history[user_to_index[user_id]] = []
        users_history[user_to_index[user_id]].append(session)
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
        else ts - last_time_done[project_to_index[project_id]]
    n_tasks = 1
    user_id = raw[1]
    last_time_done[project_to_index[project_id]] = ts
    return user_id, USession(project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


def lastfm_prepare_data(data):
    data[:, 2] = np.array(list(map(lambda x: time.mktime(time.strptime(x, "%Y-%m-%dT%H:%M:%SZ")), data[:, 2])))
    data = data[np.argsort(data[:, 2])]
    users_history = {}
    user_to_index = {}
    project_to_index = {}
    last_time_done = {}
    # make combination to session
    last_session = None
    for val in data:
        user_id, session = lastfm_raw_to_session(val, project_to_index, last_time_done)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history[user_to_index[user_id]] = []
        if last_session is not None and last_session.pid == session.pid:
            continue
        users_history[user_to_index[user_id]].append(session)
    return users_history


# common

def train_test_split(data, train_ratio):
    train, test = {}, {}
    start_times = []
    for user_id, user_history in data.items():
        train[user_id] = []
        for session in user_history:
            start_times.append(session.start_ts)
    split_time = np.percentile(np.array(start_times), train_ratio * 100)
    # print("split time", split_time)
    for user_id, user_history in data.items():
        train[user_id] = []
        test[user_id] = []
        for i, session in enumerate(user_history):
            if session.start_ts < split_time:
                train[user_id].append(session)
            else:
                test[user_id].append(session)
    return train, test


def train(data, dim, beta, other_project_importance, learning_rate, iter_num):
    model = Model2Lambda(data, dim, learning_rate=learning_rate, eps=1, beta=beta,
                         other_project_importance=other_project_importance)
    for i in range(iter_num):
        if i % 5 == 0:
            print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model.optimization_step()
    model_application = ModelApplication(model.user_embeddings, model.project_embeddings, beta,
                                         other_project_importance).fit(data)
    return model_application


def return_time_mae(model, data, samples_num=10):
    errors = 0.
    count = 0
    for user_id, user_history in data.items():
        for i, session in enumerate(user_history):
            if i > 0 and not math.isnan(session.pr_delta):
                count += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                errors += abs(expected_return_time - session.pr_delta)
            model.accept(user_id, session)
    return errors / count


def real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio):
    X_tr, X_te = train_test_split(X, train_ratio)
    model_application = train(X_tr, dim, beta, other_project_importance, learning_rate, iter_num)
    return_time = return_time_mae(model_application, X_te, samples_num=samples_num)
    print("return_time:", return_time)


def toloka_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    learning_rate = 6.5
    iter_num = 60
    size = 10000
    samples_num = 10
    train_ratio = 0.7
    raw_data = toloka_read_raw_data(TOLOKA_FILENAME, size)
    X = toloka_prepare_data(raw_data)
    print("objects_num:", size)
    print("users_num:", len(X))
    real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio)


def lastfm_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    learning_rate = 0.0005
    iter_num = 10
    size = 1000
    samples_num = 10
    train_ratio = 0.7
    raw_data = lastfm_read_raw_data(LASTFM_FILENAME, size)
    X = lastfm_prepare_data(raw_data)
    print(X)
    print("objects_num:", size)
    print("users_num:", len(X))
    real_data_test(X, dim, beta, other_project_importance, learning_rate, iter_num, samples_num, train_ratio)


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    # toloka_test()
    lastfm_test()
    print("time:", time.time() - start_time)
