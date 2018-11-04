import numpy as np


T_BEGIN = 0
T_END = 1
PROJECT_ID = 2
N_DONE = 3
EMBEDDING_DIMS = ...


def time_deltas(user):
    return (next_step[T_BEGIN] - cur_step[T_END] for next_step, cur_step in zip(user[1:], user[:-1]))


def calc_lambdas(user, user_embedding, project_embeddings, beta, *, derivative=False):
    numerator, denominator = 0., 0.
    user_d_numerator = np.zeros(EMBEDDING_DIMS)
    for cur_step, delta_time in zip(user[:-1], time_deltas(user)):
        e = np.exp(-beta * delta_time)
        numerator += e * cur_step[N_DONE] * (user_embedding @ project_embeddings[cur_step[PROJECT_ID]])
        denominator += e
        if derivative:
            user_d_numerator += e * cur_step[N_DONE] * project_embeddings[cur_step[PROJECT_ID]]
            project_d_numerator = e * cur_step[N_DONE] * user_embedding
            yield numerator / denominator, user_d_numerator / denominator, project_d_numerator / denominator
        else:
            yield numerator / denominator


def log_likelihood(users, project_embeddings, user_embeddings, beta, eps):
    ll = 0.
    for user, user_embedding in zip(users, user_embeddings):
        for lam, delta_time in zip(calc_lambdas(user, user_embedding, project_embeddings, beta), time_deltas(user)):
            ll += np.log(-(np.exp(-lam * (delta_time + eps)) - np.exp(-lam * (delta_time - eps))) / lam)
    return ll


def calc_derivative(users, project_embeddings, user_embeddings, beta, eps):
    users_derivatives = []
    project_derivatives = [np.zeros(EMBEDDING_DIMS) for _ in project_embeddings]
    for user, user_embedding in zip(users, user_embeddings):
        user_d = 0.
        for cur_step, next_step, (lam, lam_usr_d, lam_proj_d) in zip(user[:-1], user[1:],
                calc_lambdas(user, user_embedding, project_embeddings, beta, derivative=True)):
            delta_time = next_step[T_BEGIN] - cur_step[T_END]
            cur_ll_d = -lam / (np.exp(-lam * (delta_time + eps)) - np.exp(-lam * (delta_time - eps))) * \
                   ((1 / lam + delta_time + eps) * np.exp(-lam * (delta_time + eps))
                    - (1 / lam + delta_time - eps) * np.exp(-lam * (delta_time - eps)))
            user_d += cur_ll_d * lam_usr_d
            project_derivatives[cur_step[PROJECT_ID]] += cur_ll_d * lam_proj_d
        users_derivatives.append(user_d)
    return users_derivatives, project_derivatives
