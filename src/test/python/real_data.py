import time
import math
import numpy as np
import pandas as pd
from src.main.python.model import USession, Model2Lambda, ModelApplication


FILENAME = ""


def toloka_read_raw_data(filename, size=None):
    raw_data = pd.read_json(filename, lines=True).values
    return raw_data if size is None else raw_data[:size]


def toloka_raw_to_session(raw, project_to_index, start_time=0):
    project_id = raw[3]
    start_ts = (raw[4] - start_time) / (60 * 60)
    end_ts = (raw[0] - start_time) / (60 * 60)
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
    start_time = np.min(data[:, 4])
    for val in data:
        user_id, session = toloka_raw_to_session(val, project_to_index, start_time)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history[user_to_index[user_id]] = []
        users_history[user_to_index[user_id]].append(session)
    return users_history


def train_test_split(data, train_ratio):
    train, test = {}, {}
    start_times = []
    for user_id, user_history in data.items():
        train[user_id] = []
        for session in user_history:
            start_times.append(session.start_ts)
    split_time = np.percentile(np.array(start_times), train_ratio * 100)
    print("split time", split_time)
    for user_id, user_history in data.items():
        train[user_id] = []
        test[user_id] = []
        for i, session in enumerate(user_history):
            if session.start_ts < split_time:
                train[user_id].append(session)
            else:
                test[user_id].append(session)

    # for user_id, user_history in data.items():
    #     for i, session in enumerate(user_history):
    #         if session.start_ts < split_time:
    #             if user_id not in train:
    #                 train[user_id] = []
    #                 test[user_id] = []
    #             train[user_id].append(session)
    #         else:
    #             if user_id in train and user_id not in test:
    #                 pass
    #             if user_id in train:
    #                 test[user_id].append(session)

    # for user_id, user_history in data.items():
    #     for i, session in enumerate(user_history):
    #         if session.start_ts < split_time:
    #             if user_id not in train:
    #                 train[user_id] = []
    #                 # test[user_id] = []
    #             train[user_id].append(session)
    # for user_id, user_history in data.items():
    #     for i, session in enumerate(user_history):
    #         if session.start_ts >= split_time:
    #             if user_id in train:
    #                 if user_id not in test:
    #                     test[user_id] = []
    #                 test[user_id].append(session)

    # keys = set(test.keys()) | set(train.keys())
    # for user_id in keys:
    #     if train[user_id] == [] or test[user_id] == []:
    #         del train[user_id]
    #         del test[user_id]
    return train, test


def train(data, dim, beta, other_project_importance, learning_rate, iter_num):
    print(len(data))
    # print(len(data[0]))
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
        if user_id % 100 == 0:
            print(user_id, "/", len(data))
        for i, session in enumerate(user_history):
            if i > 0 and not math.isnan(session.pr_delta):
                count += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                errors += abs(expected_return_time - session.pr_delta)
            model.accept(user_id, session)
    return errors / count


def real_data_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    learning_rate = 2.5
    iter_num = 30
    size = 1000
    samples_num = 10
    train_ratio = 0.7
    train_size = int(size * train_ratio)
    raw_data = toloka_read_raw_data(FILENAME, size)
    X = toloka_prepare_data(raw_data)
    X_tr, X_te = train_test_split(X, train_ratio)
    print(X_tr)
    print(X_te)
    model_application = train(X_tr, dim, beta, other_project_importance, learning_rate, iter_num)
    return_time = return_time_mae(model_application, X_te, samples_num=samples_num)
    print(return_time)


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    real_data_test()
    print("time:", time.time() - start_time)
