import copy
import numpy as np
from collections import namedtuple

now_ts = 1541019601  # 2018-11-01 00:00:01

USession = namedtuple('USession', 'pid start_ts end_ts pr_delta n_tasks')


class UserProjectLambda:
    def __init__(self, user_embedding, beta, square=False):
        self.numerator = 1
        self.denominator = 1
        self.beta = beta
        self.square = square
        self.user_embedding = user_embedding
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.project_d_numerators = np.zeros(user_embedding.shape)

    def update(self, project_embedding, n_tasks, delta_time, coeff, derivative=True):
        e = np.exp(-self.beta * delta_time)
        ua = self.user_embedding @ project_embedding
        self.numerator = e * self.numerator + coeff * n_tasks * ua * (ua if self.square else 1)
        self.denominator = e * self.denominator + coeff
        if derivative:
            self.user_d_numerator = e * self.user_d_numerator + coeff * n_tasks * \
                                    project_embedding * (2 * ua if self.square else 1)
            self.project_d_numerators = e * self.project_d_numerators + coeff * n_tasks * \
                                        self.user_embedding * (2 * ua if self.square else 1)

    def get(self, derivative=False):
        cur_lambda = self.numerator / self.denominator
        if derivative:
            user_derivative = self.user_d_numerator / self.denominator
            project_derivative = self.project_d_numerators / self.denominator
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class UserLambda:
    def __init__(self, user_embedding, project_embeddings, beta, other_project_importance, square=False):
        self.project_lambdas = {}
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding
        self.project_embeddings = project_embeddings
        self.default_lambda = UserProjectLambda(self.user_embedding, self.beta, square)

    def update(self, session, delta_time, derivative=True):
        if session.pid not in self.project_lambdas:
            self.project_lambdas[session.pid] = copy.deepcopy(self.default_lambda)
        for project_id in self.project_lambdas.keys():
            coeff = 1 if project_id == session.pid else self.other_project_importance
            self.project_lambdas[session.pid].update(self.project_embeddings[project_id], session.n_tasks, delta_time,
                                                     coeff, derivative)
            self.default_lambda.update(self.project_embeddings[project_id], session.n_tasks, delta_time,
                                       self.other_project_importance, derivative)

    def get(self, project_id, derivative=False):
        if project_id not in self.project_lambdas.keys():
            return self.default_lambda.get(derivative)
        return self.project_lambdas[project_id].get(derivative)


class Model:
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5):
        self.users_history = users_history
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.eps = eps
        self.type = 0
        self.other_project_importance = other_project_importance

        self.user_embeddings = np.array(
            [np.random.normal(0, 0.51, self.emb_dim) for _ in range(len(self.users_history))])
        projects_set = set()
        for user_history in self.users_history:
            for session in user_history:
                projects_set.add(session.pid)
        self.project_embeddings = np.array([np.random.normal(0, 0.51, self.emb_dim) for _ in range(len(projects_set))])
        # self.init_user_embeddings()
        # self.init_project_embeddings()

    def init_user_embeddings(self):
        self.user_embeddings = np.array([np.random.normal(0, 1, self.emb_dim) for _ in range(len(self.users_history))])

    def init_project_embeddings(self):
        projects_set = set()
        for user_history in self.users_history:
            for session in user_history:
                projects_set.add(session.pid)
        self.project_embeddings = np.array([np.random.normal(0, 1, self.emb_dim) for _ in range(len(projects_set))])

    def log_likelihood(self):
        return self._likelihood_derivative()

    def ll_derivative(self):
        return self._likelihood_derivative(derivative=True)

    def _likelihood_derivative(self, derivative=False):
        ll = 0.
        users_derivatives = np.zeros(self.user_embeddings.shape)
        project_derivatives = np.zeros(self.project_embeddings.shape)
        for user_id in range(len(self.users_history)):
            user_history = self.users_history[user_id]
            user_lambda = UserLambda(self.user_embeddings[user_id], self.project_embeddings,
                                     self.beta, self.other_project_importance, self.type == 0)
            done_projects = set()
            last_times_sessions = set()
            for i, user_session in enumerate(user_history):
                # first time done projects tasks, skip ll update
                if user_session.pid not in done_projects:
                    done_projects.add(user_session.pid)
                    if i > 0:
                        user_lambda.update(user_session, user_session.start_ts - user_history[i - 1].start_ts)
                    continue

                if user_session.n_tasks != 0:
                    if derivative:
                        self._update_session_derivative(user_session, user_id, user_lambda, users_derivatives,
                                                        project_derivatives)
                    else:
                        ll += self._update_session_likelihood(user_session, user_lambda)
                    user_lambda.update(user_session, user_session.start_ts - user_history[i - 1].start_ts)
                else:
                    last_times_sessions.add(user_session)

            for user_session in last_times_sessions:
                if derivative:
                    self._update_last_derivative(user_session, user_id, user_lambda, users_derivatives,
                                                 project_derivatives)
                else:
                    ll += self._update_last_likelihood(user_session, user_lambda)
        if derivative:
            return users_derivatives, project_derivatives
        return ll

    # def _session_likelihood(self, user_session, user_lambda):
    #     cur_lambda = user_lambda.get(user_session.pid)
    #     return np.log(-(np.exp(-cur_lambda * (user_session.pr_delta + self.eps)) -
    #                    np.exp(-cur_lambda * (user_session.pr_delta - self.eps))) / cur_lambda)
    #
    # def _last_likelihood(self, user_session, user_lambda):
    #     cur_lambda = user_lambda.get(user_session.pid)
    #     return np.log(1. + (np.exp(-cur_lambda * user_session.pr_delta) - 1) / cur_lambda)
    #
    # def _session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
    #     lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
    #     tau = user_session.pr_delta
    #     cur_ll_d = -((1 / lam + tau + self.eps) * np.exp(-lam * (tau + self.eps)) -
    #                  (1 / lam + tau - self.eps) * np.exp(-lam * (tau - self.eps))) / \
    #                 (np.exp(-lam * (tau + self.eps)) - np.exp(-lam * (tau - self.eps)))
    #     users_derivatives[user_id] += cur_ll_d * lam_user_d
    #     project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d
    #
    # def _last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
    #     lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
    #     tau = user_session.pr_delta
    #     cur_ll_d = ((tau + 1 / lam) * np.exp(-lam * tau) - 1 / lam) / (lam + np.exp(-lam * tau) - 1)
    #     users_derivatives[user_id] -= cur_ll_d * lam_user_d
    #     project_derivatives[user_session.pid] -= cur_ll_d * lam_projects_d

    def optimization_step(self):
        users_derivatives, project_derivatives = self.ll_derivative()
        for i in range(len(self.user_embeddings)):
            self.user_embeddings[i] += self.learning_rate * users_derivatives[i]
        for i in range(len(self.project_embeddings)):
            self.project_embeddings[i] += self.learning_rate * project_derivatives[i]

    def get_user_embeddings(self):
        return self.user_embeddings

    def get_project_embeddings(self):
        return self.project_embeddings


class Model2UA(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance)
        self.square = False

    def _update_session_likelihood(self, user_session, user_lambda):
        cur_lambda = user_lambda.get(user_session.pid)
        return np.log(-np.exp(-cur_lambda * (user_session.pr_delta + self.eps)) +
                      np.exp(-cur_lambda * (user_session.pr_delta - self.eps)))

    def _update_last_likelihood(self, user_session, user_lambda):
        return -user_lambda.get(user_session.pid) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        tau = user_session.pr_delta
        cur_ll_d = ((tau + self.eps) * np.exp(-lam * (tau + self.eps)) -
                    (tau - self.eps) * np.exp(-lam * (tau - self.eps))) / \
                   (-np.exp(-lam * (tau + self.eps)) + np.exp(-lam * (tau - self.eps)))
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        users_derivatives[user_id] -= user_session.pr_delta * lam_user_d
        project_derivatives[user_session.pid] -= user_session.pr_delta * lam_projects_d


class Model2Lambda(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance)
        self.square = False

    def _update_session_likelihood(self, user_session, user_lambda):
        cur_lambda2 = user_lambda.get(user_session.pid) ** 2
        return np.log(-np.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) +
                      np.exp(-cur_lambda2 * (user_session.pr_delta - self.eps)))

    def _update_last_likelihood(self, user_session, user_lambda):
        return -(user_lambda.get(user_session.pid) ** 2) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        lam2 = lam ** 2
        tau = user_session.pr_delta
        cur_ll_d = 2 * lam * ((tau + self.eps) * np.exp(-lam2 * (tau + self.eps)) -
                              (tau - self.eps) * np.exp(-lam2 * (tau - self.eps))) / \
                   (-np.exp(-lam2 * (tau + self.eps)) + np.exp(-lam2 * (tau - self.eps)))
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        users_derivatives[user_id] -= 2 * lam * user_session.pr_delta * lam_user_d
        project_derivatives[user_session.pid] -= 2 * lam * user_session.pr_delta * lam_projects_d
