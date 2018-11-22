import copy
import math
import numpy as np
from collections import namedtuple
import warnings


USession = namedtuple('USession', 'pid start_ts end_ts pr_delta n_tasks')


class UserProjectLambda:
    def __init__(self, user_embedding, n_projects, beta, derivative=False, square=False):
        self.numerator = 10
        self.denominator = 10
        self.beta = beta
        # self.square = square
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.project_d_numerators = np.zeros((n_projects, len(user_embedding)))
        self.avg_time_between_sessions = 5

    def update(self, project_embedding, project_id, n_tasks, delta_time, coeff):
        e = np.exp(-self.beta)  # * delta_time)
        ua = self.user_embedding @ project_embedding
        self.numerator = e * self.numerator + coeff * ua * n_tasks  # * (ua if self.square else 1)
        self.denominator = e * self.denominator + coeff
        if self.derivative:
            self.user_d_numerator = e * self.user_d_numerator + coeff * project_embedding * n_tasks  # * (2 * ua if self.square else 1)
            self.project_d_numerators *= e
            if project_id is not None:
                self.project_d_numerators[project_id] += coeff * self.user_embedding * n_tasks  # * (2 * ua if self.square else 1)

    def get(self):
        cur_lambda = self.numerator / self.denominator / self.avg_time_between_sessions
        if self.derivative:
            user_derivative = self.user_d_numerator / self.denominator / self.avg_time_between_sessions
            project_derivative = self.project_d_numerators / self.denominator / self.avg_time_between_sessions
            if math.isnan(user_derivative[0]):
                print(self.denominator)
            if math.isnan(user_derivative[0]):
                print(self.user_d_numerator)
            # if math.isnan(project_derivative[0]):
            #     print(self.project_d_numerators)
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class UserLambda:
    def __init__(self, user_embedding, n_projects, beta, other_project_importance,
                 derivative=False, square=False):
        self.project_lambdas = {}
        self.beta = beta
        self.derivative = derivative
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding
        if math.isnan(user_embedding[0]):
            print("u_e is nan")
        self.default_lambda = UserProjectLambda(self.user_embedding, n_projects, self.beta, derivative, square)

    def update(self, project_embedding, session, delta_time):
        if session.pid not in self.project_lambdas:
            self.project_lambdas[session.pid] = copy.deepcopy(self.default_lambda)
        for project_id in self.project_lambdas.keys():
            coeff = 1 if project_id == session.pid else self.other_project_importance
            self.project_lambdas[project_id].update(project_embedding, project_id, session.n_tasks,
                                                    delta_time, coeff)
        self.default_lambda.update(project_embedding, None, session.n_tasks,
                                   delta_time, self.other_project_importance)

    def get(self, project_id):
        if project_id not in self.project_lambdas.keys():
            return self.default_lambda.get()
        return self.project_lambdas[project_id].get()


class Model:
    def __init__(self, users_history, dim, learning_rate=0.003, beta=0.001, eps=3600, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        self.users_history = users_history
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = 0.97
        self.beta = beta
        self.eps = eps
        self.data_size = sum([len(user) for user in users_history])
        self.other_project_importance = other_project_importance

        self.user_embeddings = np.array(
            [np.random.normal(0, 0.2, self.emb_dim) for _ in range(len(self.users_history))]) \
            if users_embeddings_prior is None else users_embeddings_prior
        if projects_embeddings_prior is None:
            projects_set = set()
            for user_history in self.users_history:
                for session in user_history:
                    projects_set.add(session.pid)
            self.project_embeddings = np.array(
                [np.random.normal(0, 0.2, self.emb_dim) for _ in range(len(projects_set))])
        else:
            self.project_embeddings = projects_embeddings_prior
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
            user_lambda = UserLambda(self.user_embeddings[user_id], len(project_derivatives), self.beta,
                                     self.other_project_importance, derivative, self.square)
            done_projects = set()
            last_times_sessions = set()
            for i, user_session in enumerate(user_history):
                # first time done projects tasks, skip ll update
                if user_session.pid not in done_projects:
                    done_projects.add(user_session.pid)
                    if i > 0:
                        user_lambda.update(self.project_embeddings[user_session.pid], user_session,
                                           user_session.start_ts - user_history[i - 1].start_ts)
                    continue

                if user_session.n_tasks != 0:
                    if derivative:
                        self._update_session_derivative(user_session, user_id, user_lambda, users_derivatives,
                                                        project_derivatives)
                    else:
                        ll += self._session_likelihood(user_session, user_lambda)
                    user_lambda.update(self.project_embeddings[user_session.pid], user_session,
                                       user_session.start_ts - user_history[i - 1].start_ts)
                else:
                    last_times_sessions.add(user_session)

            for user_session in last_times_sessions:
                if derivative:
                    self._update_last_derivative(user_session, user_id, user_lambda, users_derivatives,
                                                 project_derivatives)
                else:
                    ll += self._last_likelihood(user_session, user_lambda)
        if derivative:
            if math.isnan(users_derivatives[0][0]) or math.isnan(project_derivatives[0][0]):
                print(users_derivatives)
            return users_derivatives, project_derivatives
        return ll

    def _session_likelihood(self, user_session, user_lambda):
        raise NotImplementedError()

    def _last_likelihood(self, user_session, user_lambda):
        raise NotImplementedError()

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def optimization_step(self):
        users_derivatives, project_derivatives = self.ll_derivative()
        if math.isnan(users_derivatives[0][0]):
            print(users_derivatives)
        self.user_embeddings += users_derivatives * self.learning_rate / self.data_size
        self.project_embeddings += project_derivatives * self.learning_rate / self.data_size
        self.learning_rate *= self.decay_rate


class Model2UA(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior)
        self.square = True

    def _session_likelihood(self, user_session, user_lambda):
        cur_lambda = user_lambda.get(user_session.pid)
        return np.log(-np.exp(-cur_lambda * (user_session.pr_delta + self.eps)) +
                      np.exp(-cur_lambda * (user_session.pr_delta - self.eps)))

    def _last_likelihood(self, user_session, user_lambda):
        return -user_lambda.get(user_session.pid) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        tau = user_session.pr_delta
        cur_ll_d = ((tau + self.eps) * np.exp(-lam * (tau + self.eps)) -
                    max(0, tau - self.eps) * np.exp(-lam * max(0, tau - self.eps))) / \
                   (-np.exp(-lam * (tau + self.eps)) + np.exp(-lam * max(0, tau - self.eps)))
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        users_derivatives[user_id] -= user_session.pr_delta * lam_user_d
        project_derivatives[user_session.pid] -= user_session.pr_delta * lam_projects_d


class Model2Lambda(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior)
        self.square = False

    def _session_likelihood(self, user_session, user_lambda):
        cur_lambda2 = user_lambda.get(user_session.pid) ** 2
        if math.isnan(cur_lambda2):
            print("Is None")
        val = -np.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) + np.exp(
            -cur_lambda2 * max(0, user_session.pr_delta - self.eps))
        if not (0 <= val <= 1):
            print(val)
            assert 0 <= val <= 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = np.log(-np.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) +
                      np.exp(-cur_lambda2 * max(0, user_session.pr_delta - self.eps)))
            if w and w[0].category == RuntimeWarning:
                ans = -1e6
                warnings.warn("in ll", RuntimeWarning)
        return ans

    def _last_likelihood(self, user_session, user_lambda):
        return -(user_lambda.get(user_session.pid) ** 2) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid)
        lam2 = lam ** 2
        tau = user_session.pr_delta
        exp_plus = np.exp(-lam2 * (tau + self.eps))
        exp_minus = np.exp(-lam2 * max(0, tau - self.eps))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                        -exp_plus + exp_minus)
            if w and w[0].category == RuntimeWarning:
                cur_ll_d = 0
                warnings.warn("in derivative", RuntimeWarning)

        if math.isnan(cur_ll_d):
            cur_ll_d = 0
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives += cur_ll_d * lam_projects_d
        if math.isnan(users_derivatives[0][0]) or math.isnan(project_derivatives[0][0]):
            print(users_derivatives)

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid)
        users_derivatives[user_id] -= 2 * lam * user_session.pr_delta * lam_user_d
        project_derivatives -= 2 * lam * user_session.pr_delta * lam_projects_d


class ModelExpLambda(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance)
        self.square = False

    def _session_likelihood(self, user_session, user_lambda):
        exp_lambda = np.exp(user_lambda.get(user_session.pid))
        if math.isnan(exp_lambda):
            print("Is None")
        val = -np.exp(-exp_lambda * (user_session.pr_delta + self.eps)) + np.exp(
            -exp_lambda * max(0, user_session.pr_delta - self.eps))
        if not (0 <= val <= 1):
            print(val)
            assert 0 <= val <= 1
        return np.log(-np.exp(-exp_lambda * (user_session.pr_delta + self.eps)) +
                      np.exp(-exp_lambda * max(0, user_session.pr_delta - self.eps)))

    def _last_likelihood(self, user_session, user_lambda):
        return -(np.exp(user_lambda.get(user_session.pid))) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        exp_lam = np.exp(lam)
        tau = user_session.pr_delta
        cur_ll_d = exp_lam * ((tau + self.eps) * np.exp(-exp_lam * (tau + self.eps)) -
                              max(0, tau - self.eps) * np.exp(-exp_lam * max(0, tau - self.eps))) / \
                   (-np.exp(-exp_lam * (tau + self.eps)) + np.exp(-exp_lam * max(0, tau - self.eps)))
        if cur_ll_d > 100:
            print(cur_ll_d, tau)
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        users_derivatives[user_id] -= np.exp(lam) * user_session.pr_delta * lam_user_d
        project_derivatives[user_session.pid] -= np.exp(lam) * user_session.pr_delta * lam_projects_d
