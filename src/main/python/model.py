import math
import warnings

from collections import namedtuple
import numpy as np

from src.main.python.lambda_calc import UserLambda, LazyInteractionsCalculator, SimpleInteractionsCalculator,\
    UserProjectLambdaManagerLookAhead
from src.test.python.metrics import return_time_mae, item_recommendation_mae


USession = namedtuple('USession', 'pid start_ts end_ts pr_delta n_tasks')


def print_metrics(model_application, X_te, samples_num=10):
    return_time = return_time_mae(model_application, X_te, samples_num=samples_num)
    recommend_mae = item_recommendation_mae(model_application, X_te)
    print("return_time = {}, recommendation_mae = {}".format(return_time, recommend_mae))


def sort_data_by_time(users_history):
    events = []
    for user_id, user_history in users_history.items():
        for session in user_history:
            events.append((user_id, session))
    return sorted(events, key=lambda x: x[1].start_ts)


class Model:
    def __init__(self, users_histories, dim, learning_rate=0.003, beta=0.001, eps=3600, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None, square=False):
        self.users_histories = users_histories
        self.events = sort_data_by_time(users_histories)
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = 1  # 0.95
        self.beta = beta
        self.eps = eps
        self.square = square
        self.data_size = sum([len(user_history) for user_history in users_histories.values()])
        self.other_project_importance = other_project_importance

        self.user_embeddings = {user_id: np.abs(np.random.normal(0.1, 0.1, self.emb_dim))
                                for user_id, _ in self.users_histories.items()} \
            if users_embeddings_prior is None else users_embeddings_prior
        if projects_embeddings_prior is None:
            self.project_embeddings = {}
            for user_history in self.users_histories.values():
                for session in user_history:
                    if session.pid not in self.project_embeddings:
                        self.project_embeddings[session.pid] = np.abs(np.random.normal(0.1, 0.1, self.emb_dim))
        else:
            self.project_embeddings = projects_embeddings_prior

        self.default_lambda = 1.
        self.lambda_confidence = 1.
        pr_deltas = []
        for user_history in self.users_histories.values():
            for session in user_history:
                if session.pr_delta is not None and not math.isnan(session.pr_delta):
                    pr_deltas.append(session.pr_delta)
        pr_deltas = np.array(pr_deltas)
        self.default_lambda = 1 / np.mean(pr_deltas)
        print("default_lambda", self.default_lambda)

        # self.project_indices = {user_id: projects_index(history) for user_id, history in self.users_histories.items()}
        # self.reversed_project_indices = {user_id: reverse_projects_indices(index)
        #                                  for user_id, index in self.project_indices.items()}
        # self.history_for_lambda = [convert_history(history, reversed_projects_index)
        #                            for history, reversed_projects_index
        #                            in zip(self.users_histories, self.reversed_project_indices)]

    def log_likelihood(self):
        return self._likelihood_derivative()

    def ll_derivative(self):
        return self._likelihood_derivative(derivative=True)

    def _likelihood_derivative(self, derivative=False):
        ll = 0.
        users_derivatives = {k: np.zeros_like(v) for k, v in self.user_embeddings.items()}
        project_derivatives = {k: np.zeros_like(v) for k, v in self.project_embeddings.items()}
        done_projects = set()
        last_times_sessions = set()
        lambdas_by_project = UserProjectLambdaManagerLookAhead(
            self.user_embeddings, self.project_embeddings, self.beta, self.other_project_importance,
            self.default_lambda, self.lambda_confidence, derivative, self.square
        )
        for user_id, session in self.events:
            # first time done projects tasks, skip ll update
            # i forget why we do like this
            if session.pid not in done_projects:
                done_projects.add(session.pid)
                lambdas_by_project.accept(user_id, session)
                continue

            if session.n_tasks != 0:
                if derivative:
                    self._update_session_derivative(user_id, session, lambdas_by_project, users_derivatives,
                                                    project_derivatives)
                else:
                    ll += self._session_likelihood(user_id, session, lambdas_by_project)
                lambdas_by_project.accept(user_id, session)
            else:
                last_times_sessions.add((user_id, session))

        for user_id, session in last_times_sessions:
            if derivative:
                self._update_last_derivative(user_id, session, lambdas_by_project, users_derivatives,
                                             project_derivatives)
            else:
                ll += self._last_likelihood(user_id, session, lambdas_by_project)
        if derivative:
            return users_derivatives, project_derivatives
        return ll

    def glove_like_optimisation(self, iter_num=30, verbose=False, eval=None):
        lr = self.learning_rate
        self.learning_rate *= self.decay_rate
        discount_decay = 0.99
        # we should check, that python change embeddings everywhere while optimizing
        users_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.user_embeddings.items()}
        # users_diffs_squares = np.ones(self.user_embeddings.shape)  # * 1e-5
        projects_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.project_embeddings.items()}
        # projects_diffs_squares = np.ones(self.project_embeddings.shape)  # * 1e-5
        interaction_calculator = SimpleInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        for optimization_iter in range(iter_num):
            for user_id, user_history in self.users_histories.items():
                user_lambda = UserLambda(self.user_embeddings[user_id], self.beta, self.other_project_importance,
                                         interaction_calculator.get_user_supplier(user_id),
                                         default_lambda=self.default_lambda, lambda_confidence=self.lambda_confidence,
                                         derivative=True, square=self.square)
                done_projects = set()
                last_times_sessions = set()
                lambdas_by_project = {}
                for i, user_session in enumerate(user_history):
                    # first time done projects tasks, skip ll update
                    if user_session.pid not in done_projects:
                        done_projects.add(user_session.pid)
                        user_lambda.update(self.project_embeddings[user_session.pid], user_session,
                                           user_session.start_ts - user_history[i - 1].start_ts)
                        lambdas_by_project[user_session.pid] = user_lambda.get(user_session.pid, accum=False)
                        continue

                    if user_session.n_tasks != 0:
                        # update in session
                        lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
                        # lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, accum=False)
                        lam2 = lam ** 2
                        tau = user_session.pr_delta
                        exp_plus = np.exp(-lam2 * (tau + self.eps))
                        exp_minus = np.exp(-lam2 * max(0, tau - self.eps))
                        cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                                -exp_plus + exp_minus)
                        if math.isnan(cur_ll_d):
                            cur_ll_d = 0.

                        user_diff = cur_ll_d * lam_user_d
                        users_diffs_squares[user_id] = discount_decay * users_diffs_squares[user_id] + user_diff * user_diff
                        self.user_embeddings[user_id] += user_diff * lr / np.sqrt(users_diffs_squares[user_id])
                        for project_id in lam_projects_d:
                            project_diff = cur_ll_d * lam_projects_d[project_id]
                            projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[project_id] + project_diff * project_diff
                            self.project_embeddings[project_id] += project_diff * lr / np.sqrt(projects_diffs_squares[project_id])

                        user_lambda.update(self.project_embeddings[user_session.pid], user_session,
                                           user_session.start_ts - user_history[i - 1].start_ts)
                        lambdas_by_project[user_session.pid] = user_lambda.get(user_session.pid, accum=False)
                    else:
                        last_times_sessions.add(user_session)

                for user_session in last_times_sessions:
                    # update in the end
                    # lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, accum=False)
                    lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
                    user_diff = 2 * lam * user_session.pr_delta * lam_user_d
                    users_diffs_squares[user_id] = discount_decay * users_diffs_squares[user_id] + user_diff * user_diff
                    self.user_embeddings[user_id] -= user_diff * lr / np.sqrt(users_diffs_squares[user_id])
                    for project_id in lam_projects_d:
                        project_diff = 2 * lam * user_session.pr_delta * lam_projects_d[project_id]
                        projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[project_id] +  project_diff * project_diff
                        self.project_embeddings[project_id] -= project_diff * lr / np.sqrt(projects_diffs_squares[project_id])
            if verbose:  # and (optimization_iter % 5 == 0 or optimization_iter in [1, 2]):
                print("{}-th iter, ll = {}".format(optimization_iter, self.log_likelihood()))
                if eval is not None:
                    print_metrics(self.get_applicable(), eval, samples_num=10)
                print()

    def _session_likelihood(self, user_id, user_session, lambdas_by_project):
        raise NotImplementedError()

    def _last_likelihood(self, user_id, user_session, lambdas_by_project):
        raise NotImplementedError()

    def _update_session_derivative(self, user_id, user_session, lambdas_by_project, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def _update_last_derivative(self, user_id, user_session, lambdas_by_project, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def get_applicable(self):
        # TODO: decide something with lambda square
        return ApplicableModel(self.user_embeddings, self.project_embeddings, self.beta, self.other_project_importance,
                               self.default_lambda).fit(self.users_histories)

    def optimization_step(self):
        users_derivatives, project_derivatives = self.ll_derivative()
        lr = self.learning_rate / self.data_size
        for user_id in self.user_embeddings:
            self.user_embeddings[user_id] += users_derivatives[user_id] * lr
        for project_id in self.project_embeddings:
            self.project_embeddings[project_id] += project_derivatives[project_id] * lr
        self.learning_rate *= self.decay_rate


class Model2UA(Model):
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = True)

    def _session_likelihood(self, user_id, user_session, user_lambda):
        cur_lambda = user_lambda.get(user_session.pid)
        return np.log(-np.exp(-cur_lambda * (user_session.pr_delta + self.eps)) +
                      np.exp(-cur_lambda * (user_session.pr_delta - self.eps)))

    def _last_likelihood(self, user_id, user_session, user_lambda):
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
                       users_embeddings_prior, projects_embeddings_prior, square=False)

    def _session_likelihood(self, user_id, user_session, lambdas_by_project):
        cur_lambda2 = lambdas_by_project.get(user_id, user_session.pid) ** 2
        # cur_lambda2 = lambdas_by_project[user_session.pid] ** 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = np.log(-np.exp(-cur_lambda2 * (user_session.pr_delta + self.eps)) +
                         np.exp(-cur_lambda2 * max(0, user_session.pr_delta - self.eps)))
            if w and w[0].category == RuntimeWarning:
                ans = 0  # -1e6
                warnings.warn("in ll", RuntimeWarning)
        return ans

    def _last_likelihood(self, user_id, user_session, lambdas_by_project):
        return -(lambdas_by_project.get(user_id, user_session.pid) ** 2) * user_session.pr_delta
        # return -(lambdas_by_project[user_session.pid] ** 2) * user_session.pr_delta

    def _update_session_derivative(self, user_id, user_session, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(user_id, user_session.pid)
        # lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
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
        # cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
        #         -exp_plus + exp_minus)
        users_derivatives[user_id] += cur_ll_d * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id]
        # if math.isnan(users_derivatives[0][0]) or math.isnan(project_derivatives[0][0]):
        #    print(users_derivatives)

    def _update_last_derivative(self, user_id, user_session, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(user_id, user_session.pid)
        # lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
        coeff = 2 * lam * user_session.pr_delta
        users_derivatives[user_id] -= coeff * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] -= coeff * lam_projects_d[project_id]


class ModelExpLambda(Model):
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = False)
        self.square = False

    def _session_likelihood(self, user_id, user_session, user_lambda):
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

    def _last_likelihood(self, user_id, user_session, user_lambda):
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


class ApplicableModel:
    def __init__(self, user_embeddings, project_embeddings, beta, other_project_importance, default_lambda,
                 lambda_transform=lambda x: x ** 2):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.beta = beta
        self.lambda_transform = lambda_transform
        self.other_project_importance = other_project_importance
        self.interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        self.user_lambdas = {user_id: UserLambda(user_embedding, self.beta,
                                                 self.other_project_importance,
                                                 self.interaction_calculator.get_user_supplier(user_id),
                                                 default_lambda=default_lambda, lambda_confidence=1, derivative=False)
                             for user_id, user_embedding in user_embeddings.items()}
        self.default_user_id = list(user_embeddings.keys())[0]
        self.last_user_session = {}
        self.lambda_values = {user_id: {} for user_id in user_embeddings}

    def accept(self, user_id, session):
        if user_id not in self.user_lambdas or session.pid not in self.project_embeddings:
            # what we can do?
            return
        if user_id in self.last_user_session:
            self.user_lambdas[user_id].update(self.project_embeddings[session.pid], session,
                                              session.start_ts - self.last_user_session[user_id].start_ts)
            self.lambda_values[user_id][session.pid] = self.lambda_transform(self.user_lambdas[user_id].get(session.pid))
        self.last_user_session[user_id] = session

    def get_lambda(self, user_id, project_id):
        # fix default lambda
        if user_id not in self.user_lambdas or project_id not in self.project_embeddings or \
                user_id not in self.lambda_values or project_id not in self.lambda_values[user_id]:
            return self.lambda_transform(self.user_lambdas.get(user_id, self.user_lambdas[self.default_user_id])
                                         .get(project_id if project_id in self.project_embeddings else -1))
        return self.lambda_values[user_id][project_id]

    def time_delta(self, user_id, project_id, size=1):
        lam = self.get_lambda(user_id, project_id)
        return np.mean(np.random.exponential(scale=1 / lam, size=size))

    def fit(self, users_history):
        events = sort_data_by_time(users_history)
        for user_id, session in events:
            self.accept(user_id, session)
        return self
