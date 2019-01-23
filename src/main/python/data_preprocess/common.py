import random

import numpy as np


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


def filter_data(data, top=False, users_num=None, projects_num=None):
    users_stat, projects_stat = top_data(data) if top else random_data(data)
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
        # it's wrong!
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
