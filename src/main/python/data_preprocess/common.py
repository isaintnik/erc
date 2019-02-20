import random

import numpy as np


def interaction_matrix(users, projects):
    users = np.array(list(users.values()))
    projects = np.array(list(projects.values()))
    return users @ projects.T


def top_data(data, key):
    count_stat = {}
    for event in data:
        if key(event) not in count_stat:
            count_stat[key(event)] = 0
        count_stat[key(event)] += 1
    count_stat = list(count_stat.items())
    count_stat = sorted(count_stat, key=lambda x: -x[1])
    return [key for key, value in count_stat]


def random_data(data, key):
    count_stat = set()
    for event in data:
        count_stat.add(key(event))
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
    for event in data:
        if event.uid in selected_users and event.pid in selected_projects:
            new_data.append(event)
            new_users.add(event.uid)
            new_projects.add(event.pid)
    print("After filtering: |Events| = {}, |users| = {}, |projects| = {}".format(len(new_data), len(new_users),
                                                                                 len(new_projects)))
    return new_data


def get_split_time(data, train_ratio):
    start_times = []
    for event in data:
        start_times.append(event.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


# def train_test_split(data, train_ratio):
#     train, test = [], []
#     data = sorted(data, key=lambda s: s.start_ts)
#     split_time = get_split_time(data, train_ratio)
#     active_time = {}
#     seen_users = set()
#     seen_projects = set()
#     skipped_projects = set()
#     for event in data:
#         # it's wrong!
#         # if event.uid not in seen_users and event.pid not in seen_projects and event.start_ts >= split_time:
#         #     skipped_projects.add(event.pid)
#         #     continue
#         # seen_users.add(event.uid)
#         # seen_projects.add(event.pid)
#         if event.uid not in active_time:
#             active_time[event.uid] = [np.inf, -np.inf]
#         active_time[event.uid][0] = min(active_time[event.uid][0], event.start_ts)
#         active_time[event.uid][1] = max(active_time[event.uid][1], event.start_ts)
#
#     for event in data:
#         # it's wrong!
#         # if event.uid not in seen_users and event.pid not in seen_projects and event.start_ts >= split_time:
#         #     skipped_projects.add(event.pid)
#         #     continue
#         # seen_users.add(event.uid)
#         # seen_projects.add(event.pid)
#         if active_time[event.uid][0] < split_time:
#             if event.start_ts < split_time:
#                 train.append(event)
#             else:
#                 test.append(event)
#     return train, test


def train_test_split(data, train_ratio):
    train, test = [], []
    data = sorted(data, key=lambda s: s.start_ts)
    split_time = get_split_time(data, train_ratio)
    seen_projects = set()
    seen_users = set()
    for event in data:
        if event.start_ts > split_time and (event.pid not in seen_projects or event.uid not in seen_users):
            continue
        seen_projects.add(event.pid)
        seen_users.add(event.uid)
        if event.start_ts < split_time:
            train.append(event)
        else:
            test.append(event)
    return train, test
