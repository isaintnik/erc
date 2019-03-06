import math
import random
import numpy as np


def return_time_mae(model, events):
    errors = 0.
    count = 0
    for event in events:
        if event.pr_delta is not None and not math.isnan(event.pr_delta) and event.n_tasks > 0:
            count += 1
            expected_return_time = model.time_delta(event.uid, event.pid)
            # expected_return_time = 0
            errors += abs(expected_return_time - event.pr_delta)
            model.accept(event)
    return errors / count


def item_recommendation_mae(model, events):
    errors = 0.
    count = 0
    for event in events:
        if event.n_tasks > 0:
            if event.uid not in model.user_embeddings or event.pid not in model.project_embeddings:
                count -= 1
                continue
            project_lambdas = [(model.get_lambda(event.uid, pid), pid)
                               for pid in model.project_embeddings]
            sorted_pids = [pid for l, pid in sorted(project_lambdas, reverse=True)]
            errors += sorted_pids.index(event.pid)
            count += 1

            model.accept(event)
    return errors / count


def constant_prediction_time_mae(train, test):
    pr_deltas = np.array([event.pr_delta for event in train if event.pr_delta is not None and event.n_tasks != 0])
    expected_return_time = np.mean(pr_deltas)
    errors = 0.
    count = 0
    for event in test:
        if event.pr_delta is not None and not math.isnan(event.pr_delta) and event.n_tasks > 0:
            count += 1
            errors += abs(expected_return_time - event.pr_delta)
    return errors / count


def unseen_recommendation(model, train, test, top=1):
    will_see_projects = {}
    all_unseen_projects = {user_id: set(model.project_embeddings.keys()) for user_id in model.user_embeddings if str(user_id) != "-1"}

    for event in test:
        if event.uid not in will_see_projects:
            will_see_projects[event.uid] = set()
        if event.uid not in all_unseen_projects:
            all_unseen_projects[event.uid] = set()
        will_see_projects[event.uid].add(event.pid)
        all_unseen_projects[event.uid].add(event.pid)

    for event in train:
        if event.uid in will_see_projects:
            will_see_projects[event.uid].discard(event.pid)
        if event.uid in all_unseen_projects:
            all_unseen_projects[event.uid].discard(event.pid)
    match = 0.
    count = len(test)
    skipped = 0
    for event in test:
        if event.uid not in model.user_embeddings or event.pid not in model.project_embeddings:
            skipped += 1
            continue
        project_lambdas = [(model.get_lambda(event.uid, pid), pid)
                           for pid in all_unseen_projects[event.uid]]
        top_projects = [pid for l, pid in sorted(project_lambdas, reverse=True)][:top]
        ok = False
        for project in top_projects:
            if project in will_see_projects[event.uid]:
                ok = True
                break
        if ok:
            match += 1
        will_see_projects[event.uid].discard(event.pid)
        model.accept(event)
    # print("top = {}, count = {}, skipped = {}".format(top, count, skipped))
    return match / (count - skipped)


def unseen_recommendation_random(model, train, test, top=1):
    will_see_projects = {}
    all_unseen_projects = {user_id: set(model.project_embeddings.keys()) for user_id in model.user_embeddings if str(user_id) != "-1"}

    for event in test:
        if event.uid not in will_see_projects:
            will_see_projects[event.uid] = set()
        if event.uid not in all_unseen_projects:
            all_unseen_projects[event.uid] = set()
        will_see_projects[event.uid].add(event.pid)
        all_unseen_projects[event.uid].add(event.pid)

    for event in train:
        if event.uid in will_see_projects:
            will_see_projects[event.uid].discard(event.pid)
        if event.uid in all_unseen_projects:
            all_unseen_projects[event.uid].discard(event.pid)
        # if event.uid in will_see_projects and event.pid in will_see_projects[event.uid]:
        #     will_see_projects[event.uid].discard(event.pid)
        #     all_unseen_projects[event.uid].discard(event.pid)
    match = 0.
    count = len(test)
    skipped = 0
    for event in test:
        if event.uid not in model.user_embeddings or event.pid not in model.project_embeddings:
            skipped += 1
            continue
        project_lambdas = [(random.random(), pid)
                           for pid in all_unseen_projects[event.uid]]
        top_projects = [pid for l, pid in sorted(project_lambdas, reverse=True)][:top]
        ok = False
        for project in top_projects:
            if project in will_see_projects[event.uid]:
                ok = True
                break
        if ok:
            match += 1
        will_see_projects[event.uid].discard(event.pid)
        model.accept(event)
    return match / (count - skipped)


def print_metrics(model, train_data, test_data):
    test_return_time = return_time_mae(model.get_applicable(train_data), test_data)
    train_return_time = return_time_mae(model.get_applicable(), train_data)
    recommend_mae = item_recommendation_mae(model.get_applicable(train_data), test_data)
    print(f"test_return_time = {test_return_time}, train_return_time = {train_return_time}, "
          f"recommendation_mae = {recommend_mae}")
