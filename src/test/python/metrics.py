import math


def return_time_mae(model, events, samples_num=10):
    errors = 0.
    count = 0
    for session in events:
        if session.pr_delta is not None and not math.isnan(session.pr_delta) and session.n_tasks > 0:
            count += 1
            expected_return_time = model.time_delta(session.uid, session.pid, samples_num)
            # expected_return_time = 0
            errors += abs(expected_return_time - session.pr_delta)
        model.accept(session)
    return errors / count


def item_recommendation_mae(model, events):
    errors = 0.
    count = 0
    for session in events:
        if session.n_tasks > 0:
            observed_next_pid = session.pid
            if observed_next_pid not in model.project_embeddings:
                continue

            project_lambdas = [(model.user_lambdas[session.uid].get(pid), pid)
                               for pid in model.project_embeddings]
            sorted_pids = [pid for l, pid in sorted(project_lambdas, reverse=True)]
            errors += sorted_pids.index(observed_next_pid)
            count += 1

            model.accept(session)
    return errors / count


def unseen_recommendation(model, train, test):
    unseen_projects = {user_id: set() for user_id in model.user_embeddings}
    for session in test:
        unseen_projects[session.uid].add(session.pid)
    for session in train:
        if session.pid in unseen_projects[session.uid]:
            unseen_projects[session.uid].add(session.pid)
    for session in test:
        pass
