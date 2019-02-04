import random

import numpy as np


def interaction_matrix(users, projects):
    users = np.array(list(users.values()))
    projects = np.array(list(projects.values()))
    return users @ projects.T


def top_data(data, key):
    count_stat = {}
    for session in data:
        if key(session) not in count_stat:
            count_stat[key(session)] = 0
        count_stat[key(session)] += 1
    count_stat = list(count_stat.items())
    count_stat = sorted(count_stat, key=lambda x: -x[1])
    return [key for key, value in count_stat]


def random_data(data, key):
    count_stat = set()
    for session in data:
        count_stat.add(key(session))
    count_stat = list(count_stat)
    random.shuffle(count_stat)
    return count_stat


def filter_data(data, top=False, users_num=None, projects_num=None):
    if top:
        users_stat = top_data(data, lambda x: x.uid)
        projects_stat = top_data(data, lambda x: x.pid)
    else:
        users_stat = random_data(data, lambda x: x.uid)
        projects_stat = random_data(data, lambda x: x.pid)
    users_num = len(users_stat) if users_num is None else min(users_num, len(users_stat))
    projects_num = len(projects_stat) if projects_num is None else min(projects_num, len(projects_stat))
    selected_users = set(users_stat[:users_num])
    selected_projects = set(projects_stat[:projects_num])

    new_data = []
    new_users = set()
    new_projects = set()
    for session in data:
        if session.uid in selected_users and session.pid in selected_projects:
            new_data.append(session)
            new_users.add(session.uid)
            new_projects.add(session.pid)
    print("After filtering: |Events| = {}, |users| = {}, |projects| = {}".format(len(new_data), len(new_users),
                                                                                 len(new_projects)))
    return new_data


def get_split_time(data, train_ratio):
    start_times = []
    for session in data:
        start_times.append(session.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


def train_test_split(data, train_ratio):
    train, test = [], []
    data = sorted(data, key=lambda s: s.start_ts)
    split_time = get_split_time(data, train_ratio)
    active_time = {}
    seen_users = set()
    seen_projects = set()
    skipped_projects = set()
    for session in data:
        # it's wrong!
        # if session.uid not in seen_users and session.pid not in seen_projects and session.start_ts >= split_time:
        #     skipped_projects.add(session.pid)
        #     continue
        # seen_users.add(session.uid)
        # seen_projects.add(session.pid)
        if session.uid not in active_time:
            active_time[session.uid] = [np.inf, -np.inf]
        active_time[session.uid][0] = min(active_time[session.uid][0], session.start_ts)
        active_time[session.uid][1] = max(active_time[session.uid][1], session.start_ts)

    for session in data:
        # it's wrong!
        # if session.uid not in seen_users and session.pid not in seen_projects and session.start_ts >= split_time:
        #     skipped_projects.add(session.pid)
        #     continue
        # seen_users.add(session.uid)
        # seen_projects.add(session.pid)
        if active_time[session.uid][0] < split_time:
            if session.start_ts < split_time:
                train.append(session)
            else:
                test.append(session)
    return train, test
