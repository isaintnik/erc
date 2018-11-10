from collections import defaultdict
import numpy as np


# TODO: switch from tuples to named tuples
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


class LambdaCalculator:
    def __init__(self, beta, user_embedding, project_embeddings, derivative=False):
        self.numerator = 0
        self.denominator = 0
        self.beta = beta
        self.user_embedding = user_embedding
        self.project_embeddings = project_embeddings
        self.derivative = derivative
        self.user_d_numerator = np.zeros(len(user_embedding))
        self.project_d_numerator = None

    def update(self, cur_step, delta_time, coefficient=1.):
        e = np.exp(-self.beta * delta_time)
        self.numerator += coefficient * e * cur_step[N_DONE] * \
            (self.user_embedding @ self.project_embeddings[cur_step[PROJECT_ID]])
        self.denominator += coefficient * e
        if self.derivative:
            self.user_d_numerator += coefficient * e * cur_step[N_DONE] * self.project_embeddings[cur_step[PROJECT_ID]]
            self.project_d_numerator = coefficient * e * cur_step[N_DONE] * self.user_embedding

    def get(self):
        # TODO: find the default lambda
        cur_lambda = 0. if self.denominator == 0. else self.numerator / self.denominator
        if self.derivative:
            user_derivative = 0. if self.denominator == 0. else self.user_d_numerator / self.denominator
            project_derivative = 0. if self.denominator == 0. else self.project_d_numerator / self.denominator
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class Model:
    def __init__(self, users_history, dimensionality, learning_rate=0.01, beta=0.01, eps=3600,
                 external_lambda_coefficient=0.5):
        self.users_history = users_history
        self.embedding_dim = dimensionality
        self.user_embeddings = {ind: np.random.normal(0, 0.5, dimensionality) for
                                (ind, user) in enumerate(users_history)}
        project_count = max(
            max(project[PROJECT_ID] for project in user_history) for user_history in users_history
        )
        self.project_embeddings = [np.random.normal(0, 0.5, dimensionality) for _ in range(project_count)]
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps
        self.external_lambda_coefficient = external_lambda_coefficient

    def log_likelihood(self):
        ll = 0.
        for user_history, user_embedding in zip(self.users_history, self.user_embeddings):
            n = len(user_history)
            project_lambdas = defaultdict(LambdaCalculator)
            for i, cur_step in enumerate(user_history):
                lambda_calculator = project_lambdas[cur_step[PROJECT_ID]]
                cur_lambda = lambda_calculator.get()
                j = i + 1
                while j < n and user_history[j][PROJECT_ID] != user_history[i][PROJECT_ID]:
                    j += 1
                if j < n:
                    delta_time = user_history[j][T_BEGIN] - user_history[i][T_END]
                    ll += np.log(-(np.exp(-cur_lambda * (delta_time + self.eps)) -
                                   np.exp(-cur_lambda * (delta_time - self.eps))) / cur_lambda)
                    lambda_calculator.update(cur_step, delta_time)
                    for project_id, calculator in project_lambdas.items():
                        if project_id != cur_step[PROJECT_ID]:
                            calculator.update(cur_step, delta_time, self.external_lambda_coefficient)
                else:
                    # TODO: use the time to logs snapshot
                    ll += np.log(1 + (np.exp(cur_lambda * ...) - 1) / cur_lambda)
        return ll

    def calc_derivative(self):
        users_derivatives = []
        project_derivatives = [np.zeros(self.embedding_dim) for _ in self.project_embeddings]
        for user_history, user_embedding in zip(self.users_history, self.user_embeddings):
            user_d = 0.
            for (project_t_delta, project_id), (lam, lam_user_d, lam_project_d) in \
                    zip(project_time_deltas(user_history),
                        calc_lambdas(user_history, user_embedding, self.project_embeddings, self.beta,
                                     derivative=True)):
                cur_ll_d = -lam / (
                        np.exp(-lam * (project_t_delta + self.eps)) - np.exp(-lam * (project_t_delta - self.eps))) \
                           * ((1 / lam + project_t_delta + self.eps) * np.exp(-lam * (project_t_delta + self.eps))
                              - (1 / lam + project_t_delta - self.eps) * np.exp(
                            -lam * (project_t_delta - self.eps)))
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
