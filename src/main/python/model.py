import copy
import math
from collections import namedtuple
import warnings
import multiprocessing
import operator
import numpy as np
import torch


USession = namedtuple('USession', 'pid start_ts end_ts pr_delta n_tasks')


_EXECUTOR = None


def _get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = multiprocessing.Pool()
    return _EXECUTOR


def interaction_matrix(users, projects):
    return users @ projects.t()


class InteractionCalculator:
    def __init__(self, user_embeddings, project_embeddings, device='cpu'):
        self.interactions = interaction_matrix(user_embeddings.to(device), project_embeddings.to(device)).cpu()

    def get_interaction(self, user_id, project_id):
        return self.interactions[user_id, project_id].item()

    def get_user_supplier(self, user_id):
        return lambda project_id: self.interactions[user_id, project_id].item()


class UserProjectLambda:
    def __init__(self, user_embedding, n_projects, beta, interactions, *, derivative=False, square=False):
        self.numerator = torch.tensor(10, dtype=torch.float64)
        self.denominator = torch.tensor(10, dtype=torch.float64)
        self.beta = beta
        # self.square = square
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.user_d_numerator = torch.zeros_like(user_embedding)
        self.project_d_numerators = torch.zeros(n_projects, len(user_embedding), dtype=torch.float64)
        self.avg_time_between_sessions = torch.tensor(5, dtype=torch.float64)
        self.interactions = interactions
        self.lambda_project_id = None
        self.e = torch.exp(torch.tensor(-beta, dtype=torch.float64))

    def set_project_id(self, project_id):
        self.lambda_project_id = project_id

    def update(self, project_embedding, session_project_id, n_tasks, delta_time, coeff):
        e = self.e
        # e = np.exp(-self.beta)  # * delta_time)
        ua = self.interactions[session_project_id]
        # ua = self.interactions_supplier(session_project_id)
        coeff_n_tasks = torch.tensor(coeff * n_tasks, dtype=torch.float64)
        self.numerator = e * self.numerator + ua * coeff_n_tasks  # * (ua if self.square else 1)
        self.denominator = e * self.denominator + coeff
        if self.derivative:
            self.user_d_numerator = self.e * self.user_d_numerator + project_embedding * coeff_n_tasks  # * (2 * ua if self.square else 1)
            self.project_d_numerators *= self.e
            if self.lambda_project_id is not None:
                self.project_d_numerators[self.lambda_project_id] += self.user_embedding * coeff_n_tasks  # * (2 * ua if self.square else 1)

    def get(self):
        div = self.denominator * self.avg_time_between_sessions
        cur_lambda = self.numerator / div
        if self.derivative:
            user_derivative = self.user_d_numerator / div
            project_derivative = self.project_d_numerators / div
            # if math.isnan(user_derivative[0]):
            #     print(self.denominator)
            # if math.isnan(user_derivative[0]):
            #     print(self.user_d_numerator)
            # if math.isnan(project_derivative[0]):
            #     print(self.project_d_numerators)
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class UserLambda:
    def __init__(self, user_embedding, n_projects, beta, other_project_importance, interactions,
                 derivative=False, square=False):
        self.project_lambdas = {}
        self.beta = beta
        self.derivative = derivative
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding
        if math.isnan(user_embedding[0]):
            print("u_e is nan")
        self.default_lambda = UserProjectLambda(self.user_embedding, n_projects, self.beta, interactions,
                                                derivative=derivative, square=square)

    def update(self, project_embedding, session, delta_time):
        if session.pid not in self.project_lambdas:
            self.project_lambdas[session.pid] = copy.deepcopy(self.default_lambda)
            self.project_lambdas[session.pid].set_project_id(session.pid)
        for project_id, project_lambda in self.project_lambdas.items():
            coefficient = 1 if project_id == session.pid else self.other_project_importance
            project_lambda.update(project_embedding, session.pid, session.n_tasks, delta_time, coefficient)
        self.default_lambda.update(project_embedding, session.pid, session.n_tasks,
                                   delta_time, self.other_project_importance)

    def get(self, project_id):
        if project_id not in self.project_lambdas.keys():
            return self.default_lambda.get()
        return self.project_lambdas[project_id].get()


class Model:
    def __init__(self, users_history, dim, learning_rate=0.003, beta=0.001, eps=3600, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None, square=False, device='cpu'):
        self.users_history = users_history
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = 0.97
        self.beta = beta
        self.eps = eps
        self.data_size = sum([len(user) for user in users_history])
        self.other_project_importance = other_project_importance
        self.square = square
        self.device = device

        self.user_embeddings = users_embeddings_prior if users_embeddings_prior is not None \
            else torch.randn(len(self.users_history), self.emb_dim) * 0.2
        if projects_embeddings_prior is None:
            projects_set = set()
            for user_history in self.users_history:
                for session in user_history:
                    projects_set.add(session.pid)
            self.project_embeddings = torch.randn(len(projects_set), self.emb_dim) * 0.2
        else:
            self.project_embeddings = projects_embeddings_prior

    def log_likelihood(self):
        return self._log_likelihood()

    def ll_derivative(self):
        return self._log_likelihood(derivative=True)

    def _usr_ll(self, args):
        derivative, interaction_calculator, user_id = args
        ll = torch.tensor(0., dtype=torch.float64)
        users_derivatives = torch.zeros_like(self.user_embeddings)
        project_derivatives = torch.zeros_like(self.project_embeddings)

        user_history = self.users_history[user_id]
        user_lambda = UserLambda(self.user_embeddings[user_id], len(project_derivatives), self.beta,
                                 self.other_project_importance, interaction_calculator.interactions[user_id],
                                 # self.other_project_importance, interaction_calculator.get_user_supplier(user_id),
                                 derivative, self.square)
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
        return ll, users_derivatives, project_derivatives

    def _log_likelihood(self, derivative=False):
        interaction_calculator = InteractionCalculator(self.user_embeddings, self.project_embeddings, self.device)

        n_users = len(self.users_history)
        # usr_res = list(map(self._usr_ll, zip([derivative] * n_users, [interaction_calculator] * n_users,
        usr_res = list(_get_executor().map(self._usr_ll, zip([derivative] * n_users, [interaction_calculator] * n_users,
                                                        range(n_users))))

        if derivative:
            users_derivatives = sum(map(operator.itemgetter(1), usr_res))
            project_derivatives = sum(map(operator.itemgetter(2), usr_res))

            if math.isnan(users_derivatives[0][0]) or math.isnan(project_derivatives[0][0]):
                print(users_derivatives)
            return users_derivatives, project_derivatives
        ll = sum(map(operator.itemgetter(0), usr_res))
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
                 users_embeddings_prior=None, projects_embeddings_prior=None, device='cpu'):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square=True, device=device)

    def _session_likelihood(self, user_session, user_lambda):
        cur_lambda = user_lambda.get(user_session.pid)
        return torch.log(-torch.exp(-cur_lambda * (user_session.pr_delta + self.eps)) +
                         torch.exp(-cur_lambda * (user_session.pr_delta - self.eps)))

    def _last_likelihood(self, user_session, user_lambda):
        return -user_lambda.get(user_session.pid) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        tau = user_session.pr_delta
        cur_ll_d = ((tau + self.eps) * torch.exp(-lam * (tau + self.eps)) -
                    max(0, tau - self.eps) * torch.exp(-lam * max(0, tau - self.eps))) / \
                   (-torch.exp(-lam * (tau + self.eps)) + torch.exp(-lam * max(0, tau - self.eps)))
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        project_derivatives[user_session.pid] += cur_ll_d * lam_projects_d

    def _update_last_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, derivative=True)
        users_derivatives[user_id] -= user_session.pr_delta * lam_user_d
        project_derivatives[user_session.pid] -= user_session.pr_delta * lam_projects_d


class Model2Lambda(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None, device='cpu'):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square=False, device=device)

    def _session_likelihood(self, user_session, user_lambda):
        cur_lambda2 = user_lambda.get(user_session.pid) ** 2
        if math.isnan(cur_lambda2):
            print("Is None")
        val = -torch.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) + \
            torch.exp(-cur_lambda2 * max(0, user_session.pr_delta - self.eps))
        if not (0 <= val <= 1):
            print(val)
            assert 0 <= val <= 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = torch.log(-torch.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) +
                            torch.exp(-cur_lambda2 * max(0, user_session.pr_delta - self.eps)))
            if w and w[0].category == RuntimeWarning:
                ans = torch.tensor(-1e6, dtype=torch.float64, device=self.device)
                warnings.warn("in ll", RuntimeWarning)
        return ans

    def _last_likelihood(self, user_session, user_lambda):
        return -(user_lambda.get(user_session.pid) ** 2) * user_session.pr_delta

    def _update_session_derivative(self, user_session, user_id, user_lambda, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid)
        lam2 = lam ** 2
        tau = user_session.pr_delta
        exp_plus = torch.exp(-lam2 * (tau + self.eps))
        exp_minus = torch.exp(-lam2 * max(0, tau - self.eps))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / \
                (-exp_plus + exp_minus)
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
        lam_del = 2 * lam * user_session.pr_delta
        users_derivatives[user_id] -= lam_del * lam_user_d
        project_derivatives -= lam_del * lam_projects_d


class ModelExpLambda(Model):
    def __init__(self, users_history, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 device='cpu'):
        Model.__init__(self, users_history, dim, learning_rate, beta, eps, other_project_importance, square=False,
                       device=device)

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
