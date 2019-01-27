import random

import numpy as np


def interaction_matrix(users, projects):
    users = np.array(list(users.values()))
    projects = np.array(list(projects.values()))
    return users @ projects.T


def top_data(data):
    users_stat = {}
    projects_stat = {}

    for session in data:
        users_stat.setdefault(session.uid, 0)
        projects_stat.setdefault(session.pid, 0)
        users_stat[session.uid] += 1
        projects_stat[session.pid] += 1

    users_stat = list(users_stat.items())
    projects_stat = list(projects_stat.items())
    sorted(users_stat, key=lambda x: -x[1])
    sorted(projects_stat, key=lambda x: -x[1])
    return [uid for (uid, value) in users_stat], [pid for (pid, value) in projects_stat]


def random_data(data):
    users_stat = set()
    projects_stat = set()
    for session in data:
        users_stat.add(session.uid)
        projects_stat.add(session.pid)

    users_stat = list(users_stat)
    projects_stat = list(projects_stat)
    random.shuffle(users_stat)
    random.shuffle(projects_stat)
    return users_stat, projects_stat


def filter_data(data, top=False, users_num=None, projects_num=None):
    users_stat, projects_stat = top_data(data) if top else random_data(data)
    user_num = len(users_stat) if users_num is None else min(users_num, len(users_stat))
    projects_num = len(projects_stat) if projects_num is None else min(projects_num, len(projects_stat))
    selected_users = set(users_stat[:user_num])
    selected_projects = set(projects_stat[:projects_num])
    before_events_num, after_events_num = 0, 0
    new_data = []
    for session in data:
        if session.uid in selected_users and session.pid in selected_projects:
            new_data.append(session)
            before_events_num += 1
    print("Events num before = {}, after = {}".format(len(data), len(new_data)))
    return data


def get_split_time(data, train_ratio):
    start_times = []
    for session in data:
        start_times.append(session.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = [], []
    split_time = get_split_time(data, train_ratio)
    seen_users = set()
    for session in data:
        # it's wrong!
        if session.uid not in seen_users and session.start_ts >= split_time:
            continue
        seen_users.add(session.uid)
        if session.start_ts < split_time:
            train.append(session)
        else:
            test.append(session)
    return train, test
