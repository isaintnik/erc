import time
import pandas as pd
from src.main.python.model import *
from src.main.python.data_generator import *


FILENAME = ""


def read_data(filename):
    df = pd.read_json(filename, lines=True)
    # pid start_ts end_ts pr_delta n_tasks
    print(df.columns)  # 'end_ts', 'inner_delta', 'n_tasks', 'project_id', 'start_ts', 'worker_id'
    vals = df.values
    X = [USession(val[3], val[4], val[0], val[1], val[2]) for val in vals]
    return X[:1000]


def real_data_test():
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    learning_rate = 0.7
    iter_num = 36
    X = read_data(FILENAME)
    model = Model2Lambda(X, dim, learning_rate=learning_rate, eps=20, beta=beta,
                         other_project_importance=other_project_importance)
    # for i in range(iter_num):
    #     if i % 5 == 0:
    #         print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
    #     model.optimization_step()

    # we can't compute whole matrix
    # end_interaction = interaction_matrix(model.user_embeddings, model.project_embeddings)
    # print(end_interaction)


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    real_data_test()
    print("time:", time.time() - start_time)
