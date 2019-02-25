import argparse
import math
import pickle

from src.test.python.real_data import make_data
from src.main.python.model import Event
from collections import defaultdict


def sort_events_by_error(model, events):
    events_by_error = []
    project_visits = defaultdict(lambda: 1)
    for event in events:
        if event.pr_delta is not None and not math.isnan(event.pr_delta) and event.n_tasks > 0:
            expected_return_time = model.time_delta(event.uid, event.pid)
            error = abs(expected_return_time - event.pr_delta)
            events_by_error.append((event, error, project_visits[event.pid]))
            project_visits[event.pid] += 1
            model.accept(event)
    events_by_error.sort(key=lambda x: -x[1])
    return events_by_error


def main(model_path, data_path):
    size = 100 * 1000
    train_ratio = 0.75
    users_num = 1000
    projects_num = 1000
    top_items = True
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    X_tr, X_te = make_data(data_path, size, users_num, projects_num, top_items, train_ratio)
    events_by_error = sort_events_by_error(model.get_applicable(X_tr), X_te)
    min_ts = min(i.start_ts for i in X_te)
    max_ts = max(i.start_ts for i in X_te)
    shift = -min_ts
    scale = 1 / (max_ts - min_ts)
    scaled_times = [
        (Event(event.uid, event.pid, (event.start_ts + shift) * scale, event.pr_delta, event.n_tasks), error, event_id)
        for event, error, event_id in events_by_error
    ]
    print(scaled_times)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('data')
    args = parser.parse_args()
    main(args.model, args.data)
