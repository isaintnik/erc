import pandas as pd

from src.main.python.model import Event


def toloka_read_raw_data(filename, size=None):
    raw_datas = pd.read_json(filename, lines=True, chunksize=size)
    for raw_data in raw_datas:
        print("original data shape", raw_data.shape)
        return raw_data


def toloka_raw_to_session(raw, last_time_done):
    project_id = raw.project_id
    start_ts = int(raw.start_ts) / (60 * 60)
    user_id = raw.worker_id
    if user_id not in last_time_done:
        last_time_done[user_id] = {}

    pr_delta = None if project_id not in last_time_done[user_id] \
        else (start_ts - last_time_done[user_id][project_id])
    n_tasks = 1
    last_time_done[user_id][project_id] = start_ts
    return Event(user_id, project_id, start_ts, pr_delta, n_tasks)


def toloka_prepare_data(data):
    events = []
    users_set = set()
    projects_set = set()
    last_time_done = {}
    for row in data.itertuples():
        session = toloka_raw_to_session(row, last_time_done)
        users_set.add(session.uid)
        projects_set.add(session.pid)
        events.append(session)
    print("Read |Events| = {}, |users| = {}, |projects| = {}".format(len(events), len(users_set), len(projects_set)))
    return events
