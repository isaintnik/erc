import math

import numpy as np
import pandas as pd

from src.main.python.model import USession


def toloka_read_raw_data(filename, size=None):
    raw_data = pd.read_json(filename, lines=True)
    print(raw_data.head())
    raw_data = raw_data.values
    print(raw_data[0, 0], raw_data[0, 1], raw_data[0, 2])
    return raw_data if size is None else raw_data[:size]


def toloka_raw_to_session(raw, user_to_index, project_to_index):
    # project_id = raw[3]
    # start_ts = raw[4] / (60 * 60)
    # end_ts = raw[0] / (60 * 60)
    # pr_delta = raw[1] / (60 * 60)
    # n_tasks = raw[2]
    # user_id = raw[5]

    project_id = raw[0]
    start_ts = int(raw[1]) / (60 * 60)
    user_id = raw[2]
    end_ts = None
    pr_delta = None
    n_tasks = 1
    if project_id not in project_to_index:
        project_to_index[project_id] = len(project_to_index)
    if user_id not in user_to_index:
        user_to_index[user_id] = len(user_to_index)
    return USession(user_to_index[user_id], project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


def toloka_prepare_data(data):
    events = []
    user_to_index = {}
    project_to_index = {}
    pr_deltas = []
    for val in data:
        session = toloka_raw_to_session(val, user_to_index, project_to_index)
        events.append(session)
        if session.pr_delta is not None and not math.isnan(session.pr_delta):
            pr_deltas.append(session.pr_delta)
    pr_deltas = np.array(pr_deltas)
    print("mean pr_delta", np.mean(pr_deltas), np.std(pr_deltas))
    return events
