from collections import defaultdict, Counter
from typing import NamedTuple

USER_ID = "user_id"
TS_START = "ts_start"
TS_END = "ts_end"
PROJECT_ID = "project_id"
N_DONE = "n_done"

fields_lst = [TS_START, TS_END, PROJECT_ID, N_DONE]


class UserStep(NamedTuple):
    ts_start: int
    ts_end: int
    project_id: int
    n_done: int = 0


def log_grouper(raw_samples):
    '''

    :param raw_samples: list of UserStepSample objects
    :return: dict with keys as user id's and values as lists of user steps
    '''
    users = defaultdict(list)

    # Counter of how many tasks in project done by user
    user_pr_cnt = Counter()

    for sample in raw_samples:
        pr_id = sample.get_str_field(PROJECT_ID)
        user_id = sample.get_str_field(USER_ID)

        # Add 'n_done' field to dict
        cnt_key = user_id + "|" + pr_id
        cnt = user_pr_cnt[cnt_key]

        sample.sample_dct[N_DONE] = cnt
        user_pr_cnt[cnt_key] += 1

        # Generate step
        step_data_lst = [sample.get_field(f) for f in fields_lst]
        step = UserStep(*step_data_lst)

        users[user_id].append(step)

    return users


class UserStepSample:
    '''
    Class for parsing logs in temporary logs Format:
    field1=value1   field2=value2   field3=value3

    Class is used only until the actual log format would be discovered
    '''

    def __init__(self, sample_str):
        self.sample_str = sample_str
        self.sample_dct = {item[0]: item[1] for item in
                           [item.split("=") for item in
                            sample_str.split("\t")]}

    def get_int_field(self, field):
        return int(self.sample_dct.get(field, 0))

    def get_str_field(self, field):
        return str(self.sample_dct.get(field, 0))
