from collections import namedtuple
from src.main.python.preprocess_data import process_data_stream


Row = namedtuple('Row', 'project_id timestamp next_ts action_type')


def test_process_data_stream():
    worker_id = "wid"
    # [(project_id, timestamp, next_ts, action_type)]
    data = [
        Row("pid1", 0, 0, "res"),

        # 1,2,5
        Row("pid1", 0, 1, "ep"),
        Row("pid2", 0, 1, "ep"),
        Row("pid5", 0, 1, "ep"),

        Row("pid2", 3, 6, "res"),
        Row("pid2", 7, 11, "res"),

        # 1,2,4
        Row("pid1", 12, 13, "ep"),
        Row("pid2", 12, 13, "ep"),
        Row("pid4", 12, 13, "ep"),

        Row("pid2", 14, 17, "res"),
        Row("pid4", 17, 18, "res"),

        # 4,5
        Row("pid4", 19, 20, "ep"),
        Row("pid5", 19, 20, "ep"),

        # 1,2,5,4
        Row("pid1", 21, 22, "ep"),
        Row("pid2", 21, 22, "ep"),
        Row("pid5", 21, 22, "ep"),
        Row("pid4", 21, 22, "ep"),

        Row("pid2", 23, 24, "res"),
        Row("pid2", 25, 26, "res"),

        # 4
        Row("pid4", 28, 29, "ep"),

        Row("pid4", 29, 30, "res"),

        Row("pid4", 32, 33, "ep"),
        Row("pid2", 32, 33, "ep"),

        Row("pid4", 35, 36, "ep")
    ]
    result = list(process_data_stream(worker_id, data))
    print(result)

    # 4 sessions
    assert len(result) == 5

    # first time doing this project
    assert {'worker_id': "wid", 'project_id': "pid2", 'start_ts': 3, 'end_ts': 11,
            'n_tasks': 2, 'inner_delta': None} in result

    # second time doing this project, inner_delta = 12-11
    assert {'worker_id': "wid", 'project_id': "pid2", 'start_ts': 14, 'end_ts': 17,
            'n_tasks': 1, 'inner_delta': 1} in result

    # third time doing this project, inner_delta = 19-17 (way of time computing may be discussed), pid4 tasks skipped
    assert {'worker_id': "wid", 'project_id': "pid2", 'start_ts': 23, 'end_ts': 26,
            'n_tasks': 2, 'inner_delta': 2} in result

    # first time doing this project
    assert {'worker_id': "wid", 'project_id': "pid4", 'start_ts': 29, 'end_ts': 30,
            'n_tasks': 1, 'inner_delta': None} in result

    assert {'worker_id': 'wid', 'project_id': 'pid2', 'start_ts': 31, 'end_ts': 31,
            'n_tasks': 0, 'inner_delta': 2} in result


if __name__ == "__main__":
    test_process_data_stream()

