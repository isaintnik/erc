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


def select_users_and_projects(data, top=False, users_num=None, projects_num=None):
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
    return selected_users, selected_projects


def filter_data(data, users, projects):
    new_data = []
    new_users = set()
    new_projects = set()
    for event in data:
        if event.uid in users and event.pid in projects:
            new_data.append(event)
            new_users.add(event.uid)
            new_projects.add(event.pid)
    print(f"After filtering: |Events| = {len(new_data)}, |users| = {len(new_users)}, |projects| = {len(new_projects)}")
    # print(new_users)
    return new_data


def get_split_time(data, train_ratio):
    start_times = []
    for event in data:
        start_times.append(event.start_ts)
    return np.percentile(np.array(start_times), train_ratio * 100)


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


def split_and_filter_data(data, train_ratio, top_items, users_num, projects_num):
    X_tr, X_te = train_test_split(data, train_ratio)
    selected_users, selected_projects = select_users_and_projects(X_tr, top=top_items, users_num=users_num,
                                                                  projects_num=projects_num)
    new_users = selected_users
    new_projects = selected_projects
    first = True
    # X_tr = filter_data(X_tr, users=selected_users, projects=selected_projects)
    while first or new_users != selected_users or new_projects != selected_projects:
        first = False
        X_tr = filter_data(X_tr, users=new_users, projects=new_projects)
        X_te = filter_data(X_te, users=new_users, projects=new_projects)
        selected_users = new_users
        selected_projects = new_projects
        new_users = set(e.uid for e in X_tr) & set(e.uid for e in X_te) & selected_users
        new_projects = set(e.pid for e in X_tr) & set(e.pid for e in X_te) & selected_projects
        print("iter of filtering")
    return X_tr, X_te
