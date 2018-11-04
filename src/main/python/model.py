import numpy as np

T_BEGIN = 0
T_END = 1
PROJECT_ID = 2
N_DONE = 3


def project_time_deltas(user_history):
    last_visit = {}
    for project in user_history:
        if project[PROJECT_ID] not in last_visit:
            yield 0., project[PROJECT_ID]  # what to do there?
        else:
            yield project[T_BEGIN] - last_visit[project[PROJECT_ID]], project[PROJECT_ID]
            last_visit[project[PROJECT_ID]] = project[T_END]


def time_deltas(user):
    return (next_step[T_BEGIN] - cur_step[T_END] for next_step, cur_step in zip(user[1:], user[:-1]))


def calc_lambdas(user, user_embedding, project_embeddings, beta, *, derivative=False):
    if not user:
        if derivative:
            yield 0.
        else:
            yield 0., 0., 0.
        return
    numerator = user[0][N_DONE] * (user_embedding @ project_embeddings[user[0][PROJECT_ID]])
    denominator = 1.
    user_d_numerator = user[0][N_DONE] * project_embeddings[user[0][PROJECT_ID]]
    if derivative:
        yield numerator
    else:
        yield numerator, user_d_numerator, 0
    for cur_step, delta_time in zip(user[1:], time_deltas(user)):
        e = np.exp(-beta * delta_time)
        numerator = e * numerator + cur_step[N_DONE] * (user_embedding @ project_embeddings[cur_step[PROJECT_ID]])
        denominator = e * denominator + 1
        if derivative:
            user_d_numerator = e * user_d_numerator + cur_step[N_DONE] * project_embeddings[cur_step[PROJECT_ID]]
            project_d_numerator = e * cur_step[N_DONE] * user_embedding
            yield numerator / denominator, user_d_numerator / denominator, project_d_numerator / denominator
        else:
            yield numerator / denominator


class Model:
    def __init__(self, users_history, embedding_dim, learning_rate=0.01, beta=0.01, eps=3600):
        self.users_history = users_history
        self.embedding_dim = embedding_dim
        self.user_embeddings = {id: np.random.normal(0, 0.5, embedding_dim) for (id, user) in enumerate(users_history)}
        project_count = max(max(project[PROJECT_ID] for project in user_history)
                                for user_history in users_history)
        self.project_embeddings = [np.random.normal(0, 0.5, embedding_dim) for _ in range(project_count)]
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps

    def log_likelihood(self):
        ll = 0.
        for user, user_embedding in zip(self.users_history, self.user_embeddings):
            for lam, (delta_time, _) in zip(calc_lambdas(user, user_embedding, self.project_embeddings, self.beta),
                                            project_time_deltas(user)):
                ll += np.log(-(np.exp(-lam * (delta_time + self.eps)) - np.exp(-lam * (delta_time - self.eps))) / lam)
        return ll

    def calc_derivative(self):
        users_derivatives = []
        project_derivatives = [np.zeros(self.embedding_dim) for _ in self.project_embeddings]
        for user_history, user_embedding in zip(self.users_history, self.user_embeddings):
            user_d = 0.
            for (project_time_delta, project_id), (lam, lam_user_d, lam_project_d) in \
                    zip(project_time_deltas(user_history),
                        calc_lambdas(user_history, user_embedding, self.project_embeddings, self.beta,
                                     derivative=True)):
                cur_ll_d = -lam / (
                        np.exp(-lam * (project_time_delta + self.eps)) - np.exp(-lam * (project_time_delta - self.eps))) \
                           * ((1 / lam + project_time_delta + self.eps) * np.exp(-lam * (project_time_delta + self.eps))
                              - (1 / lam + project_time_delta - self.eps) * np.exp(
                            -lam * (project_time_delta - self.eps)))
                user_d += cur_ll_d * lam_user_d
                project_derivatives[project_id] += cur_ll_d * lam_project_d
            users_derivatives.append(user_d)
        return users_derivatives, project_derivatives

    def optimization_step(self):
        users_derivatives, project_derivatives = self.calc_derivative()
        for i in range(len(self.user_embeddings)):
            self.user_embeddings[i] += self.learning_rate * users_derivatives[i]
        for i in range(len(self.project_embeddings)):
            self.project_embeddings[i] += self.learning_rate * project_derivatives[i]

    def get_user_embeddings(self):
        return self.user_embeddings

    def get_project_embeddings(self):
        return self.project_embeddings
