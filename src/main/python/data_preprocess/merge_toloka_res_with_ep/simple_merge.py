
def process_data_stream(worker_id, rows):
    current_session_first_row = None
    for row in rows:
        if row.action_type == "ep":
            continue
        if current_session_first_row is None:
            current_session_first_row = row
            continue
        if row.project_id != current_session_first_row.project_id:
            yield {
                'worker_id': worker_id,
                'project_id': current_session_first_row.project_id,
                'start_ts': current_session_first_row.timestamp
            }
            current_session_first_row = row
    yield {
        'worker_id': worker_id,
        'project_id': current_session_first_row.project_id,
        'start_ts': current_session_first_row.timestamp
    }
