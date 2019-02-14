import numpy as np
from src.main.python.model import ApplicableModel, Event, INVALID


def select_next_event(model, expected_next_time):
    next_pid, next_ts = min(expected_next_time.items(), key=lambda x: x[1])
    expected_next_time[next_pid] = next_ts + np.random.exponential(1 / model.get_lambda(0, next_pid))
    return next_ts, next_pid


def generate_sythetic(user_embeddings, project_embeddings, beta=0.001, other_project_importance=0.1, default_lambda=1.,
                      lambda_transform=lambda x: np.exp(x), starting_point=0, max_lifetime=5e5):
    events = []
    for user_id in user_embeddings:
        model = ApplicableModel({0: user_embeddings[user_id]}, project_embeddings, beta, other_project_importance,
                                default_lambda, lambda_transform)
        last_time_done = {pid: starting_point for pid in project_embeddings if pid != INVALID}
        expected_next_time = {pid: starting_point + np.random.exponential(1 / model.get_lambda(0, pid))
                              for pid in project_embeddings if pid != INVALID}
        while True:
            next_ts, next_pid = select_next_event(model, expected_next_time)
            if next_ts > max_lifetime:
                break
            event = Event(user_id, next_pid, next_ts, next_ts - last_time_done[next_pid], 1)
            events.append(event)
            model.accept(event)
            last_time_done[next_pid] = next_ts
        for pid in last_time_done:
            events.append(Event(user_id, pid, max_lifetime, max_lifetime - last_time_done[pid], 0))
        print("user_id = {}, lambdas = {}".format(user_id, ", ".join(["({}, {})".format(pid, model.get_lambda(0, pid)) for pid in last_time_done])))
    return sorted(events, key=lambda event: event.start_ts)
