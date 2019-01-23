import numpy as np
import pandas as pd

from src.main.python.model import USession


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
