import math

import numpy as np
import pandas as pd

from src.main.python.model import USession


def toloka_read_raw_data(filename, size=None):
    raw_datas = pd.read_json(filename, lines=True, chunksize=size)
    for raw_data in raw_datas:
        print("original data shape", raw_data.shape)
        return raw_data.values


def toloka_raw_to_session(raw, last_time_done):
    # project_id = raw[3]
    # start_ts = raw[4] / (60 * 60)
    # end_ts = raw[0] / (60 * 60)
    # pr_delta = raw[1] / (60 * 60)
    # n_tasks = raw[2]
    # user_id = raw[5]

    project_id = raw[0]
    start_ts = int(raw[1]) / (60 * 60)
    user_id = raw[2]
    # if project_id not in project_to_index:
    #     project_to_index[project_id] = len(project_to_index)
    # if user_id not in user_to_index:
    #     user_to_index[user_id] = len(user_to_index)
    #     last_time_done[user_to_index[user_id]] = {}
    if user_id not in last_time_done:
        last_time_done[user_id] = {}

    end_ts = None
    pr_delta = None if project_id not in last_time_done[user_id] \
        else (start_ts - last_time_done[user_id][project_id])
    n_tasks = 1
    last_time_done[user_id][project_id] = start_ts
    return USession(user_id, project_id, start_ts, end_ts, pr_delta, n_tasks)


def toloka_prepare_data(data):
    events = []
    users_set = set()
    projects_set = set()
    last_time_done = {}
    pr_deltas = []
    for val in data:
        session = toloka_raw_to_session(val, last_time_done)
        users_set.add(session.uid)
        projects_set.add(session.pid)
        events.append(session)
        if session.pr_delta is not None and not math.isnan(session.pr_delta):
            pr_deltas.append(session.pr_delta)
    pr_deltas = np.array(pr_deltas)
    print("mean pr_delta", np.mean(pr_deltas), np.std(pr_deltas))
    print("Read |Events| = {}, |users| = {}, |projects| = {}".format(len(events), len(users_set), len(projects_set)))
    return events
