import math


def sort_data_by_time(users_history):
    events = []
    for user_id, user_history in users_history.items():
        for session in user_history:
            events.append((user_id, session))
    return sorted(events, key=lambda x: x[1].start_ts)


def return_time_mae(model, data, samples_num=10):
    events = sort_data_by_time(data)
    errors = 0.
    count = 0
    for user_id, session in events:
        if session.pr_delta is not None and not math.isnan(session.pr_delta) and session.n_tasks > 0:
            count += 1
            expected_return_time = model.time_delta(user_id, session.pid, samples_num)
            # expected_return_time = 0
            errors += abs(expected_return_time - session.pr_delta)
        model.accept(user_id, session)
    return errors / count


def item_recommendation_mae(model, data):
    events = sort_data_by_time(data)
    errors = 0.
    count = 0
    for user_id, session in events:
        if session.n_tasks > 0:
            observed_next_pid = session.pid
            if observed_next_pid not in model.project_embeddings:
                continue

            project_lambdas = [(model.user_lambdas[user_id].get(pid), pid)
                               for pid in model.project_embeddings]
            sorted_pids = [pid for l, pid in sorted(project_lambdas, reverse=True)]
            errors += sorted_pids.index(observed_next_pid)
            count += 1

            model.accept(user_id, session)
    return errors / count
