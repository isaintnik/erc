import math
import random


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
            if session.uid not in model.user_embeddings or session.pid not in model.project_embeddings:
                count -= 1
                continue
            project_lambdas = [(model.get_lambda(session.uid, pid), pid)
                               for pid in model.project_embeddings]
            sorted_pids = [pid for l, pid in sorted(project_lambdas, reverse=True)]
            errors += sorted_pids.index(session.pid)
            count += 1

            model.accept(session)
    return errors / count


def unseen_recommendation(model, train, test, top=1):
    will_see_projects = {}
    all_unseen_projects = {user_id: set(model.project_embeddings.keys()) for user_id in model.user_embeddings if user_id >= 0}

    for session in test:
        if session.uid not in will_see_projects:
            will_see_projects[session.uid] = set()
        if session.uid not in all_unseen_projects:
            all_unseen_projects[session.uid] = set()
        will_see_projects[session.uid].add(session.pid)
        all_unseen_projects[session.uid].add(session.pid)

    for session in train:
        if session.uid in will_see_projects:
            will_see_projects[session.uid].discard(session.pid)
        if session.uid in all_unseen_projects:
            all_unseen_projects[session.uid].discard(session.pid)
    match = 0.
    count = len(test)
    skipped = 0
    for session in test:
        if session.uid not in model.user_embeddings or session.pid not in model.project_embeddings:
            skipped += 1
            continue
        project_lambdas = [(model.user_embeddings[session.uid] @ model.project_embeddings[pid].T, pid)
                           for pid in all_unseen_projects[session.uid]]
        top_projects = [pid for l, pid in sorted(project_lambdas, reverse=True)][:top]
        ok = False
        for project in top_projects:
            if project in will_see_projects[session.uid]:
                ok = True
                break
        if ok:
            match += 1
        will_see_projects[session.uid].discard(session.pid)
    # print("top = {}, count = {}, skipped = {}".format(top, count, skipped))
    return match / (count - skipped)


def unseen_recommendation_random(model, train, test, top=1):
    will_see_projects = {}
    all_unseen_projects = {user_id: set(model.project_embeddings.keys()) for user_id in model.user_embeddings if user_id >= 0}

    for session in test:
        if session.uid not in will_see_projects:
            will_see_projects[session.uid] = set()
        if session.uid not in all_unseen_projects:
            all_unseen_projects[session.uid] = set()
        will_see_projects[session.uid].add(session.pid)
        all_unseen_projects[session.uid].add(session.pid)

    for session in train:
        if session.uid in will_see_projects:
            will_see_projects[session.uid].discard(session.pid)
        if session.uid in all_unseen_projects:
            all_unseen_projects[session.uid].discard(session.pid)
        # if session.uid in will_see_projects and session.pid in will_see_projects[session.uid]:
        #     will_see_projects[session.uid].discard(session.pid)
        #     all_unseen_projects[session.uid].discard(session.pid)
    match = 0.
    count = len(test)
    skipped = 0
    for session in test:
        if session.uid not in model.user_embeddings or session.pid not in model.project_embeddings:
            skipped += 1
            continue
        project_lambdas = [(random.random(), pid)
                           for pid in all_unseen_projects[session.uid]]
        top_projects = [pid for l, pid in sorted(project_lambdas, reverse=True)][:top]
        ok = False
        for project in top_projects:
            if project in will_see_projects[session.uid]:
                ok = True
                break
        if ok:
            match += 1
        will_see_projects[session.uid].discard(session.pid)
    return match / (count - skipped)
