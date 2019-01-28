import time

import numpy as np
import pandas as pd

from src.main.python.model import USession


def lastfm_read_raw_data(filename, size=None):
    raw_data = pd.read_csv(filename, sep='\t', error_bad_lines=False).values
    return raw_data if size is None else raw_data[:size]


def lastfm_raw_to_session(raw, user_to_index, project_to_index, last_time_done):
    user_id = raw[1]
    ts = raw[2]
    project_id = raw[3]
    if project_id not in project_to_index:
        project_to_index[project_id] = len(project_to_index)
    if user_id not in user_to_index:
        user_to_index[user_id] = len(user_to_index)
        last_time_done[user_to_index[user_id]] = {}

    start_ts = ts / (60 * 60)
    end_ts = 0  # don't used
    pr_delta = None if project_to_index[project_id] not in last_time_done[user_to_index[user_id]] \
        else (ts - last_time_done[user_to_index[user_id]][project_to_index[project_id]]) / (60 * 60)
    n_tasks = 1
    last_time_done[user_to_index[user_id]][project_to_index[project_id]] = ts
    return USession(user_to_index[user_id], project_to_index[project_id], start_ts, end_ts, pr_delta, n_tasks)


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
    events = []
    user_to_index = {}
    project_to_index = {}
    last_time_done = {}
    # make combination to session
    last_session = None
    pr_deltas = []
    for val in data:
        session = lastfm_raw_to_session(val, user_to_index, project_to_index, last_time_done)
        if last_session is not None and last_session.pid == session.pid:
            continue
        # users_history[user_to_index[session.uid]].append(session)
        events.append(session)
        last_session = session
        if session.pr_delta is not None:
            pr_deltas.append(session.pr_delta)
    pr_deltas = np.array(pr_deltas)
    print("Mean pr_delta = {}, std = {}".format(np.mean(pr_deltas), np.std(pr_deltas)))
    return events
