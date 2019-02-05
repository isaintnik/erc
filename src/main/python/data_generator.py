import numpy as np
from src.main.python.model import ApplicableModel, Event

def select_next_event(model, last_time_done):
    next_times = [(last_time_done[pid] + np.random.exponential(1 / model.get_lambda(0, pid)), pid) for pid in last_time_done]
    next_ts, next_pid = sorted(next_times)[0]
    return next_ts, next_pid


def generate_sythetic(user_embeddings, project_embeddings, beta=0.001, other_project_importance=0.1, default_lambda=1.,
                      lambda_transform=lambda x: np.exp(x), start_ts=0, max_lifetime=5e5):
    events = []
    for user_id in user_embeddings:
        model = ApplicableModel({0: user_embeddings[user_id]}, project_embeddings, beta, other_project_importance,
                                default_lambda, lambda_transform)
        last_time_done = {pid: start_ts for pid in model.project_embeddings}
        current_ts = start_ts
        while current_ts < max_lifetime:
            next_ts, next_pid = select_next_event(model, last_time_done)
            event = Event(user_id, next_pid, next_ts, next_ts - last_time_done[next_pid], 1)
            events.append(event)
            model.accept(event)
            last_time_done[next_pid] = next_ts
            current_ts = next_ts
        for pid in last_time_done:
            events.append(Event(user_id, pid, max_lifetime, max_lifetime - last_time_done[pid], 0))
    return sorted(events, key=lambda event: event.start_ts)
