import time
import numpy as np
import pandas as pd
from src.main.python.model import USession, Model2Lambda, ModelApplication


FILENAME = ""


def raw_to_session(raw, project_to_index, start_time=0):
    project_id = raw[3]
    start_ts = (raw[4] - start_time) / (60 * 60)
    end_ts = (raw[0] - start_time) / (60 * 60)
    pr_delta = raw[1] / (60 * 60)
    n_tasks = raw[2]
    user_id = raw[5]
    if project_id not in project_to_index:
        project_to_index[project_id] = len(project_to_index)
    return user_id, USession(project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


def read_data(data):
    users_history = []
    user_to_index = {}
    project_to_index = {}
    start_time = np.min(data[:, 4])
    for val in data:
        user_id, session = raw_to_session(val, project_to_index, start_time)
        if user_id not in user_to_index:
            user_to_index[user_id] = len(user_to_index)
            users_history.append([])
        users_history[user_to_index[user_id]].append(session)
    return users_history


def train_test_split(data, split_time):
    train, test = [], []
    for user_id, user_history in enumerate(data):
        for i, session in enumerate(user_history):
            pass


def train(data, dim, beta, other_project_importance):
    learning_rate = 5.5
    iter_num = 40
    print(len(data))
    print(len(data[0]))
    model = Model2Lambda(data, dim, learning_rate=learning_rate, eps=1, beta=beta,
                         other_project_importance=other_project_importance)
    for i in range(iter_num):
        if i % 5 == 0:
            print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model.optimization_step()
    return model.user_embeddings, model.project_embeddings


def test_return_time(model, data, samples_num=10):
    errors = 0.
    count = 0
    for user_id, user_history in enumerate(data):
        if user_id % 100:
            print(user_id, "/", len(data))
        for i, session in enumerate(user_history):
            if i > 0:
                count += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                errors += abs(expected_return_time - session.pr_delta)
            model.accept(user_id, session)
    return errors / count


def real_data_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    size = 10000
    samples_num = 10
    train_ratio = 0.7
    train_size = int(size * train_ratio)
    raw_data = pd.read_json(FILENAME, lines=True).values[:size]
    split_time = raw_data[train_size, 4]
    X = read_data(raw_data)
    user_emb, project_emb = train(X[:train_size], dim, beta, other_project_importance)
    print("trained")
    model = ModelApplication(user_emb, project_emb, beta, other_project_importance)
    test_return_time(model, X[train_size:], samples_num=samples_num)


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    real_data_test()
    print("time:", time.time() - start_time)
