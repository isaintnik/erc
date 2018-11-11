import numpy as np


# TODO: switch from tuples to named tuples
T_BEGIN = 0
T_END = 1
PROJECT_ID = 2
N_DONE = 3


class LambdaCalculator:
    def __init__(self, beta, user_embedding, project_embeddings, derivative=False):
        self.numerator = 1e-18
        self.denominator = 1e-9
        self.beta = beta
        self.user_embedding = user_embedding
        self.project_embeddings = project_embeddings
        self.derivative = derivative
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.project_d_numerators = np.zeros((len(project_embeddings), len(user_embedding)))

    def update(self, cur_step, delta_time, coefficient=1.):
        e = np.exp(-self.beta * delta_time)
        self.numerator += coefficient * e * cur_step[N_DONE] * \
            (self.user_embedding @ self.project_embeddings[cur_step[PROJECT_ID]])
        self.denominator += coefficient * e
        if self.derivative:
            self.user_d_numerator += coefficient * e * cur_step[N_DONE] * self.project_embeddings[cur_step[PROJECT_ID]]
            self.project_d_numerators[cur_step[PROJECT_ID]] += coefficient * e * cur_step[N_DONE] * self.user_embedding

    def get(self):
        # TODO: find the default lambda
        cur_lambda = self.numerator / self.denominator
        if self.derivative:
            user_derivative = self.user_d_numerator / self.denominator
            project_derivative = self.project_d_numerators / self.denominator
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class Model:
    def __init__(self, users_history, dimensionality, learning_rate=0.01, beta=0.01, eps=3600,
                 external_lambda_coefficient=0.5):
        self.users_history = users_history
        self.embedding_dim = dimensionality
        self.user_embeddings = {ind: np.random.normal(0, 1, dimensionality) for
                                (ind, user) in enumerate(users_history)}
        project_count = max(
            max(project[PROJECT_ID] for project in user_history) for user_history in users_history) + 1
        self.project_embeddings = [np.random.normal(0, 1, dimensionality) for _ in range(project_count)]
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps
        self.external_lambda_coefficient = external_lambda_coefficient

    def log_likelihood(self):
        ll = 0.
        for user_history, user_embedding in zip(self.users_history, self.user_embeddings.values()):
            n = len(user_history)
            project_lambdas = {}
            for i, cur_step in enumerate(user_history):
                if cur_step[PROJECT_ID] not in project_lambdas:
                    project_lambdas[cur_step[PROJECT_ID]] = \
                        LambdaCalculator(self.beta, user_embedding, self.project_embeddings)
                lambda_calculator = project_lambdas[cur_step[PROJECT_ID]]
                cur_lambda = lambda_calculator.get()
                j = i + 1
                while j < n and user_history[j][PROJECT_ID] != user_history[i][PROJECT_ID]:
                    j += 1
                if j < n:
                    delta_time = user_history[j][T_BEGIN] - user_history[i][T_END]
                    ll += np.log(-(np.exp(-cur_lambda * (delta_time + self.eps)) -
                                   np.exp(-cur_lambda * (delta_time - self.eps))) / cur_lambda)
                    # TODO: consider updating lambdas after the return to the project
                    lambda_calculator.update(cur_step, delta_time)
                    for project_id, calculator in project_lambdas.items():
                        if project_id != cur_step[PROJECT_ID]:
                            calculator.update(cur_step, delta_time, self.external_lambda_coefficient)
                else:
                    pass
                    # TODO: use the time to logs snapshot
                    # final_t = ...
                    # ll += np.log(1 + (np.exp(cur_lambda * final_t) - 1) / cur_lambda)
        return ll

    # TODO: consider merging with calculating lambda
    def calc_derivative(self):
        users_derivatives = []
        project_derivatives = np.zeros((len(self.project_embeddings), self.embedding_dim))
        for user_history, user_embedding in zip(self.users_history, self.user_embeddings.values()):
            user_d = np.zeros_like(user_embedding)
            n = len(user_history)
            project_lambdas = {}
            for i, cur_step in enumerate(user_history):
                if cur_step[PROJECT_ID] not in project_lambdas:
                    project_lambdas[cur_step[PROJECT_ID]] = \
                        LambdaCalculator(self.beta, user_embedding, self.project_embeddings, derivative=True)
                lambda_calculator = project_lambdas[cur_step[PROJECT_ID]]
                lam, lam_user_d, lam_projects_d = lambda_calculator.get()
                j = i + 1
                while j < n and user_history[j][PROJECT_ID] != user_history[i][PROJECT_ID]:
                    j += 1
                if j < n:
                    delta_time = user_history[j][T_BEGIN] - user_history[i][T_END]
                    cur_ll_d = -lam / \
                        (np.exp(-lam * (delta_time + self.eps)) - np.exp(-lam * (delta_time - self.eps))) * \
                        ((1 / lam + delta_time + self.eps) * np.exp(-lam * (delta_time + self.eps)) -
                            (1 / lam + delta_time - self.eps) * np.exp(-lam * (delta_time - self.eps)))
                    user_d += cur_ll_d * lam_user_d
                    project_derivatives += cur_ll_d * lam_projects_d
                    lambda_calculator.update(cur_step, delta_time)
                    for project_id, calculator in project_lambdas.items():
                        if project_id != cur_step[PROJECT_ID]:
                            calculator.update(cur_step, delta_time, self.external_lambda_coefficient)
                else:
                    pass
                    # TODO: use the time to logs snapshot
                    # final_t = ...
                    # cur_ll_d = (lam ** -2 * (np.exp(-lam * final_t) - 1) - final_t * np.exp(-lam * final_t) / lam) / \
                    #            (1 + (np.exp(lam * ...) - 1) / lam)
                    # user_d += cur_ll_d * lam_user_d
                    # project_derivatives[cur_step[PROJECT_ID]] += cur_ll_d * lam_projects_d
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
