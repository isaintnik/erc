import time

import numpy as np
import pandas as pd

from src.main.python.model import Event


def lastfm_read_raw_data(filename, size=None):
    return pd.read_csv(filename, sep='\t', error_bad_lines=False, nrows=size).values


def lastfm_raw_to_session(raw, last_time_done):
    user_id = raw[1]
    ts = raw[2]
    project_id = raw[4]
    if user_id not in last_time_done:
        last_time_done[user_id] = {}

    start_ts = ts / (60 * 60)
    pr_delta = None if project_id not in last_time_done[user_id] \
        else (ts - last_time_done[user_id][project_id]) / (60 * 60)
    n_tasks = 1
    last_time_done[user_id][project_id] = ts
    return Event(user_id, project_id, start_ts, pr_delta, n_tasks)


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
    last_time_done = {}
    users_set = set()
    projects_set = set()
    # make combination to session
    last_session = None
    pr_deltas = []
    for val in data:
        session = lastfm_raw_to_session(val, last_time_done)
        users_set.add(session.uid)
        projects_set.add(session.pid)
        if last_session is not None and last_session.pid == session.pid:
            continue
        events.append(session)
        last_session = session
        if session.pr_delta is not None:
            pr_deltas.append(session.pr_delta)
    pr_deltas = np.array(pr_deltas)
    print("Read |Events| = {}, |users| = {}, |projects| = {}".format(len(events), len(users_set), len(projects_set)))
    print("Mean pr_delta = {}, std = {}".format(np.mean(pr_deltas), np.std(pr_deltas)))
    return events
