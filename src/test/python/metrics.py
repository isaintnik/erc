import math


def return_time_mae(model, data, samples_num=10):
    errors = 0.
    count = 0
    for user_id, user_history in data.items():
        for i, session in enumerate(user_history):
            if i > 0 and session.pr_delta is not None and not math.isnan(session.pr_delta) and session.n_tasks > 0:
                count += 1
                expected_return_time = model.time_delta(user_id, session.pid, samples_num)
                # expected_return_time = 0
                errors += abs(expected_return_time - session.pr_delta)
            model.accept(user_id, session)
    return errors / count


def item_recommendation_mae(model, data):
    errors = 0.
    count = 0
    for user_id, user_history in data.items():

        user_lambdas = model.user_lambdas[user_id]

        for i, session in enumerate(user_history):
            if session.n_tasks > 0:
                count += 1
                observed_next_pid = session.pid

                project_lambdas = [(user_lambdas.get(pid), pid)
                                   for pid in range(len(model.project_embeddings))]

                sorted_pids = [pid for l, pid in sorted(project_lambdas, reverse=True)]
                errors += sorted_pids.index(observed_next_pid)

            model.accept(user_id, session)
    return errors / count
