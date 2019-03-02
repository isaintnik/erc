import argparse
import math
import pickle
import numpy as np
from matplotlib import pyplot as plt

from src.main.python.data_preprocess.lastfm import lastfm_read_raw_data, lastfm_prepare_data
from src.test.python.real_data import split_and_filter_data, LASTFM_FILENAME
from src.main.python.model import Event, ApplicableModel
from collections import defaultdict


def sort_events_by_error(model, train_events, test_events):
    events_by_error = []
    project_visits = defaultdict(lambda: defaultdict(lambda: 0))
    for event in train_events:
        if event.n_tasks > 0:
            project_visits[event.uid][event.pid] += 1
            model.accept(event)
    print("Train end")
    for event in test_events:
        if event.n_tasks > 0 and event.pr_delta is not None and not math.isnan(event.pr_delta):
            expected_return_time = model.time_delta(event.uid, event.pid)
            error = abs(expected_return_time - event.pr_delta)
            events_by_error.append((event, error, project_visits[event.uid][event.pid]))
            project_visits[event.uid][event.pid] += 1
            model.accept(event)
    events_by_error.sort(key=lambda x: -x[1])
    return events_by_error


def main(model_path, data_path):
    size = 20 * 1000
    train_ratio = 0.75
    users_num = 1
    projects_num = 1000
    top_items = True
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    raw_data = lastfm_read_raw_data(data_path, size)
    X = lastfm_prepare_data(raw_data)
    print("Raw events num:", raw_data.shape)
    X_tr, X_te = split_and_filter_data(X, train_ratio, top_items, users_num, projects_num)
    # X_tr, X_te = make_data(data_path, size, users_num, projects_num, top_items, train_ratio)
    app_model = ApplicableModel(model.user_embeddings, model.project_embeddings, model.beta,
                                model.other_project_importance, model.lambda_transform,
                                model.lambda_strategy_constructor)
    # events_by_error = sort_events_by_error(model.get_applicable(X_tr), X_tr, X_te)[:]
    events_by_error = sort_events_by_error(app_model, [], X_tr)[:]
    errors = np.array([error for (_, error, _) in events_by_error])
    print("Train:", np.mean(errors))

    events_by_error = sort_events_by_error(app_model, [], X_te)[:]
    errors = np.array([error for (_, error, _) in events_by_error])
    print(len(errors))
    print("Test:", np.mean(errors))
    # plt.plot(range(len(errors)), errors)
    rang_error = sorted([(error, rang) for (_, error, rang) in events_by_error], key=lambda x: x[1])
    plt.scatter([rang for _, rang in rang_error], [error for error, _ in rang_error], s=1)
    plt.show()
    print_scaled_events(X_te, events_by_error)


def print_scaled_events(data, events_by_error):
    min_ts = min(i.start_ts for i in data)
    max_ts = max(i.start_ts for i in data)
    shift = -min_ts
    scale = 1 / (max_ts - min_ts)
    scaled_times = [
        (Event(event.uid, event.pid, (event.start_ts + shift) * scale, event.pr_delta, event.n_tasks), error, event_id)
        for event, error, event_id in events_by_error
    ]
    for i in range(50):
        print(scaled_times[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_load', default=None)
    parser.add_argument('--data', default=LASTFM_FILENAME)
    args = parser.parse_args()
    main(args.model_load, args.data)
