

def update_projects_time(project_last_seen_ts, project_inner_delta, project_last_time_done_ts, last_time_projects,
                         current_ts):
    for project_id in last_time_projects:
        if project_id not in project_last_seen_ts:
            project_last_seen_ts[project_id] = None
        if project_id not in project_inner_delta:
            project_inner_delta[project_id] = None
        if project_id not in project_last_time_done_ts:
            project_last_time_done_ts[project_id] = None

    # both ways is okay but have some difference
    # on large data It doesn't matter which function to use
    _update_projects_time1(project_last_seen_ts, project_inner_delta, project_last_time_done_ts, last_time_projects,
                           current_ts)
    # _update_projects_time2(project_last_seen_ts, project_inner_delta, project_last_time_done_ts, last_time_projects,
    #                        current_ts)


def _update_projects_time1(project_last_seen_ts, project_inner_delta, project_last_time_done_ts, last_time_projects,
                           current_ts):
    for project_id in project_last_seen_ts:
        if project_last_seen_ts[project_id] is not None and project_last_time_done_ts[project_id] is not None:
            if project_inner_delta[project_id] is None:
                project_inner_delta[project_id] = 0
            project_inner_delta[project_id] += current_ts - max(project_last_seen_ts[project_id],
                                                                project_last_time_done_ts[project_id])
        project_last_seen_ts[project_id] = current_ts if project_id in last_time_projects else None


def _update_projects_time2(project_last_seen_ts, project_inner_delta, project_last_time_done_ts, last_time_projects,
                           current_ts):
    for project_id in project_last_seen_ts:
        if project_last_time_done_ts[project_id] is not None:
            if project_last_seen_ts[project_id] is not None:
                if project_inner_delta[project_id] is None:
                    project_inner_delta[project_id] = 0
                if project_last_seen_ts[project_id] < project_last_time_done_ts[project_id]:
                    project_inner_delta[project_id] = 0
                else:
                    project_inner_delta[project_id] += current_ts - max(project_last_seen_ts[project_id],
                                                                        project_last_time_done_ts[project_id])
            else:
                if project_inner_delta[project_id] is None:
                    project_inner_delta[project_id] = 0
        else:
            project_inner_delta[project_id] = None
        project_last_seen_ts[project_id] = current_ts if project_id in last_time_projects else None


def process_data_stream(worker_id, rows):
    project_last_seen_ts = {}
    project_last_time_done_ts = {}
    project_inner_delta = {}
    n_tasks = 0
    start_ts = None
    prev_row = None
    is_start = True
    last_main_page_rows = set()
    last_main_page_ts = None
    for row in rows:
        # start with main page only
        if is_start and row.action_type == "res":
            continue
        is_start = False

        # only one project between main pages
        if prev_row is not None and row.action_type == "res" and prev_row.action_type == "res" \
                and row.project_id != prev_row.project_id:
            continue

        if row.action_type == "ep":
            if prev_row is not None and prev_row.action_type == "res":
                yield {
                    'worker_id': worker_id,
                    'project_id': prev_row.project_id,
                    'start_ts': start_ts,
                    'end_ts': prev_row.next_ts,
                    'n_tasks': n_tasks,
                    'inner_delta': project_inner_delta[
                        prev_row.project_id] if prev_row.project_id in project_inner_delta else None
                }
                start_ts = None
                n_tasks = 0
                project_inner_delta[prev_row.project_id] = 0

            if last_main_page_ts is None or (last_main_page_ts is not None and row.timestamp == last_main_page_ts):
                last_main_page_rows.add(row.project_id)
            else:
                if last_main_page_rows:
                    update_projects_time(project_last_seen_ts, project_inner_delta, project_last_time_done_ts,
                                         last_main_page_rows, last_main_page_ts)
                    last_main_page_rows = set()
                last_main_page_rows.add(row.project_id)
            last_main_page_ts = row.timestamp

        elif row.action_type == "res":
            if last_main_page_rows:
                update_projects_time(project_last_seen_ts, project_inner_delta, project_last_time_done_ts,
                                     last_main_page_rows, last_main_page_ts)
                last_main_page_rows = set()

            n_tasks += 1
            project_last_time_done_ts[row.project_id] = row.next_ts
            if start_ts is None:
                start_ts = row.timestamp

        prev_row = row

    if prev_row is not None and prev_row.action_type == "res":
        yield {
            'worker_id': worker_id,
            'project_id': prev_row.project_id,
            'start_ts': start_ts,
            'end_ts': prev_row.next_ts,
            'n_tasks': n_tasks,
            'inner_delta': project_inner_delta[
                prev_row.project_id] if prev_row.project_id in project_inner_delta else None
        }

    if last_main_page_rows:
        update_projects_time(project_last_seen_ts, project_inner_delta, project_last_time_done_ts,
                             last_main_page_rows, last_main_page_ts)
    if prev_row is not None:
        now_ts = prev_row.next_ts + 1
    else:
        now_ts = None
    for project_id in project_inner_delta:
        if project_inner_delta[project_id] is not None:
            yield {
                'worker_id': worker_id,
                'project_id': project_id,
                'start_ts': now_ts,
                'end_ts': now_ts,
                'n_tasks': 0,
                'inner_delta': project_inner_delta[project_id]
            }
