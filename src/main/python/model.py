import math
from collections import namedtuple
import warnings
from src.main.python.lambda_calc import *


USession = namedtuple('USession', 'pid start_ts end_ts pr_delta n_tasks')


class Model:
    def __init__(self, users_histories, dim, learning_rate=0.003, beta=0.001, eps=3600, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None, square=False):
        self.users_histories = users_histories
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = 0.97
        self.beta = beta
        self.eps = eps
        self.square = square
        self.data_size = sum([len(user) for user in users_histories])
        self.other_project_importance = other_project_importance

        self.user_embeddings = np.array(
            [np.random.normal(0, 0.2, self.emb_dim) for _ in range(len(self.users_histories))]) \
            if users_embeddings_prior is None else users_embeddings_prior
        if projects_embeddings_prior is None:
            projects_set = set()
            for user_history in self.users_histories:
                for session in user_history:
                    projects_set.add(session.pid)
            self.project_embeddings = np.array(
                [np.random.normal(0, 0.2, self.emb_dim) for _ in range(len(projects_set))])
        else:
            self.project_embeddings = projects_embeddings_prior

        # self.project_indices = list(map(projects_index, self.users_histories))
        # self.reversed_project_indices = list(map(reverse_projects_indices, self.project_indices))
        # self.history_for_lambda = [convert_history(history, reversed_projects_index)
        #                            for history, reversed_projects_index
        #                            in zip(self.users_histories, self.reversed_project_indices)]

    def log_likelihood(self):
        return self._likelihood_derivative()

    def ll_derivative(self):
        return self._likelihood_derivative(derivative=True)

    def _likelihood_derivative(self, derivative=False):
        ll = 0.
        users_derivatives = np.zeros_like(self.user_embeddings)
        project_derivatives = np.zeros_like(self.project_embeddings)
        interaction_calculator = InteractionCalculator(self.user_embeddings, self.project_embeddings)
        for user_id in range(len(self.users_histories)):
            user_history = self.users_histories[user_id]
            user_lambda = UserLambda(self.user_embeddings[user_id], self.beta, self.other_project_importance,
                                     interaction_calculator.get_user_supplier(user_id), derivative, self.square)
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
            # if not derivative:
            #     print({s.pid: user_lambda.project_lambdas[s.pid].get() for s in last_times_sessions})
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
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = True)

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
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = False)

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
        lam, lam_user_d, lam_projects_d, denominator = user_lambda.get(user_session.pid)
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
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id] / denominator
        if math.isnan(users_derivatives[0][0]) or math.isnan(project_derivatives[0][0]):
            print(users_derivatives)

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d, denominator = user_lambda.get(user_session.pid)
        coeff = 2 * lam * user_session.pr_delta
        users_derivatives[user_id] -= coeff * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] -= coeff * lam_projects_d[project_id] / denominator


class ModelExpLambda(Model):
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = False)
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
