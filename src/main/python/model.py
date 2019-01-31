import math
import warnings

from collections import namedtuple
import numpy as np

from src.main.python.lambda_calc import UserLambda, LazyInteractionsCalculator, SimpleInteractionsCalculator,\
    UserProjectLambdaManagerLookAhead, UserProjectLambdaManagerNotLookAhead
from src.test.python.metrics import return_time_mae, item_recommendation_mae


USession = namedtuple('USession', 'uid pid start_ts end_ts pr_delta n_tasks')


def print_metrics(model_application, X_te, samples_num=10):
    return_time = return_time_mae(model_application, X_te, samples_num=samples_num)
    recommend_mae = item_recommendation_mae(model_application, X_te)
    print("return_time = {}, recommendation_mae = {}".format(return_time, recommend_mae))


class Model:
    def __init__(self, events, dim, learning_rate=0.003, beta=0.001, eps=3600, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None, square=False):
        self.events = events
        self.emb_dim = dim
        self.learning_rate = learning_rate
        self.decay_rate = 1  # 0.95
        self.beta = beta
        self.eps = eps
        self.square = square
        self.data_size = len(events)
        self.other_project_importance = other_project_importance

        self.user_ids = set(session.uid for session in events)
        self.project_ids = set(session.pid for session in events)
        self.user_embeddings = {user_id: np.abs(np.random.normal(0.1, 0.1, self.emb_dim))
                                for user_id in self.user_ids} \
            if users_embeddings_prior is None else users_embeddings_prior
        if projects_embeddings_prior is None:
            self.project_embeddings = {pid: np.abs(np.random.normal(0.1, 0.1, self.emb_dim)) for pid in self.project_ids}
        else:
            self.project_embeddings = projects_embeddings_prior

        self._update_default_embeddings()

        self.default_lambda = 1.
        self.lambda_confidence = 1.
        pr_deltas = []
        for session in events:
            if session.pr_delta is not None and not math.isnan(session.pr_delta):
                pr_deltas.append(session.pr_delta)
        pr_deltas = np.array(pr_deltas)
        self.default_lambda = 1 / np.mean(pr_deltas)
        print("default_lambda", self.default_lambda)

    def log_likelihood(self):
        return self._likelihood_derivative()

    def ll_derivative(self):
        return self._likelihood_derivative(derivative=True)

    def _likelihood_derivative(self, derivative=False):
        ll = 0.
        users_derivatives = {user_id: np.zeros_like(self.user_embeddings[user_id]) for user_id in self.user_ids}
        project_derivatives = {project_id: np.zeros_like(self.project_embeddings[project_id]) for project_id in self.project_ids}
        done_projects = {user_id: set() for user_id in self.user_ids}
        last_times_sessions = set()
        interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        lambdas_by_project = UserProjectLambdaManagerLookAhead(
            self.user_embeddings, self.project_embeddings, interaction_calculator, self.beta,
            self.other_project_importance, self.default_lambda, self.lambda_confidence, derivative, accum=True,
            square=self.square)
        # user_lambdas = {user_id: UserLambda(self.user_embeddings[user_id], self.beta, self.other_project_importance,
        #                          interaction_calculator.get_user_supplier(user_id),
        #                          default_lambda=self.default_lambda, lambda_confidence=self.lambda_confidence,
        #                          derivative=derivative, square=self.square) for user_id in self.user_ids}
        # lambdas_by_project = {}
        for session in self.events:
            # if session.uid not in lambdas_by_project:
            #     lambdas_by_project[session.uid] = {}

            # first time done projects tasks, skip ll update
            # i forget why we do like this
            if session.pid not in done_projects[session.uid]:
                done_projects[session.uid].add(session.pid)
                lambdas_by_project.accept(session)
                # user_lambdas[session.uid].update(self.project_embeddings[session.pid], session, 1)
                # lambdas_by_project[session.uid][session.pid] = user_lambdas[session.uid].get(session.pid)
                continue

            if session.n_tasks != 0:
                if derivative:
                    self._update_session_derivative(session, lambdas_by_project, users_derivatives,
                                                    project_derivatives)
                else:
                    ll += self._session_likelihood(session, lambdas_by_project)
                lambdas_by_project.accept(session)
                # user_lambdas[session.uid].update(self.project_embeddings[session.pid], session, 1)
                # lambdas_by_project[session.uid][session.pid] = user_lambdas[session.uid].get(session.pid)
            else:
                last_times_sessions.add(session)

        for session in last_times_sessions:
            if derivative:
                self._update_last_derivative(session, lambdas_by_project, users_derivatives,
                                             project_derivatives)
            else:
                ll += self._last_likelihood(session, lambdas_by_project)
        if derivative:
            return users_derivatives, project_derivatives
        return ll

    def glove_like_optimisation(self, iter_num=30, verbose=False, eval=None):
        self.learning_rate *= self.decay_rate
        discount_decay = 0.99
        users_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.user_embeddings.items()}
        projects_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.project_embeddings.items()}
        interaction_calculator = SimpleInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        done_projects = {user_id: set() for user_id in self.user_embeddings}
        last_times_sessions = set()
        lambdas_by_project = UserProjectLambdaManagerNotLookAhead(
            self.user_embeddings, self.project_embeddings, interaction_calculator, self.beta, self.other_project_importance,
            self.default_lambda, self.lambda_confidence, derivative=True, accum=False, square=True
        )
        for optimization_iter in range(iter_num):
            for session in self.events:
                if session.pr_delta is None or session.pid not in done_projects[session.uid]:
                    done_projects[session.uid].add(session.pid)
                    lambdas_by_project.accept(session)
                    continue
                # assert not (session.pid in done_projects[session.uid] and session.pr_delta is None)
                if session.n_tasks != 0:
                    self._update_glove_session_params(session, lambdas_by_project, users_diffs_squares,
                                                      projects_diffs_squares, discount_decay, self.learning_rate)
                    lambdas_by_project.accept(session)
                else:
                    last_times_sessions.add(session)
            for session in last_times_sessions:
                self._update_glove_last_params(session, lambdas_by_project, users_diffs_squares,
                                               projects_diffs_squares, discount_decay, self.learning_rate)
            self.learning_rate *= self.decay_rate
            if verbose:
                print("{}-th iter, ll = {}".format(optimization_iter, self.log_likelihood()))
                if eval is not None:
                    print_metrics(self.get_applicable(), eval, samples_num=10)
                print()

    def _update_glove_session_params(self, session, lambdas_by_project, users_diffs_squares,
                                     projects_diffs_squares, discount_decay, lr):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(session.uid, session.pid)
        # lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, accum=False)
        tr_lam = lam ** 2
        tau = session.pr_delta
        exp_plus = np.exp(-tr_lam * (tau + self.eps))
        exp_minus = np.exp(-tr_lam * max(0, tau - self.eps))
        cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                -exp_plus + exp_minus)
        if math.isnan(cur_ll_d):
            cur_ll_d = 0.

        user_diff = cur_ll_d * lam_user_d
        users_diffs_squares[session.uid] = discount_decay * users_diffs_squares[session.uid] + user_diff * user_diff
        self.user_embeddings[session.uid] += user_diff * lr / np.sqrt(users_diffs_squares[session.uid])
        for project_id in lam_projects_d:
            project_diff = cur_ll_d * lam_projects_d[project_id]
            projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[
                project_id] + project_diff * project_diff
            self.project_embeddings[project_id] += project_diff * lr / np.sqrt(
                projects_diffs_squares[project_id])

    def _update_glove_last_params(self, session, lambdas_by_project, users_diffs_squares,
                                  projects_diffs_squares, discount_decay, lr):
        # update in the end
        # lam, lam_user_d, lam_projects_d = user_lambda.get(user_session.pid, accum=False)
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(session.uid, session.pid)
        user_diff = 2 * lam * session.pr_delta * lam_user_d
        users_diffs_squares[session.uid] = discount_decay * users_diffs_squares[session.uid] + user_diff * user_diff
        self.user_embeddings[session.uid] -= user_diff * lr / np.sqrt(users_diffs_squares[session.uid])
        for project_id in lam_projects_d:
            project_diff = 2 * lam * session.pr_delta * lam_projects_d[project_id]
            projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[
                project_id] + project_diff * project_diff
            self.project_embeddings[project_id] -= project_diff * lr / np.sqrt(projects_diffs_squares[project_id])

    def _session_likelihood(self, session, lambdas_by_project):
        raise NotImplementedError()

    def _last_likelihood(self, session, lambdas_by_project):
        raise NotImplementedError()

    def _update_session_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def _update_last_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        raise NotImplementedError()

    def get_applicable(self):
        # TODO: decide something with lambda square
        return ApplicableModel(self.user_embeddings, self.project_embeddings, self.beta, self.other_project_importance,
                               self.default_lambda).fit(self.events)

    def optimization_step(self):
        users_derivatives, project_derivatives = self.ll_derivative()
        lr = self.learning_rate / self.data_size
        for user_id in self.user_ids:
            self.user_embeddings[user_id] += users_derivatives[user_id] * lr
        for project_id in self.project_ids:
            self.project_embeddings[project_id] += project_derivatives[project_id] * lr
        self.learning_rate *= self.decay_rate

    def _update_default_embeddings(self):
        mean_user = np.zeros_like(self.user_embeddings[0])
        mean_project = np.zeros_like(self.project_embeddings[0])
        for user_embedding in self.user_embeddings.values():
            mean_user += user_embedding
        for project_embedding in self.project_embeddings.values():
            mean_project += project_embedding
        self.user_embeddings[-1] = mean_user
        self.project_embeddings[-1] = mean_project


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

    def _session_likelihood(self, session, lambdas_by_project):
        # lam = lambdas_by_project[session.uid][session.pid]
        lam = lambdas_by_project.get(session.uid, session.pid)
        tr_lam = lam ** 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = np.log(-np.exp(-tr_lam * (session.pr_delta + self.eps)) +
                         np.exp(-tr_lam * max(0, session.pr_delta - self.eps)))
            if w and w[0].category == RuntimeWarning:
                ans = 0  # -1e6
                warnings.warn("in ll", RuntimeWarning)
        # ans = np.log(-np.exp(-tr_lam * (session.pr_delta + self.eps)) +
        #              np.exp(-tr_lam * max(0, session.pr_delta - self.eps)))
        return ans

    def _last_likelihood(self, session, lambdas_by_project):
        # lam = lambdas_by_project[session.uid][session.pid]
        lam = lambdas_by_project.get(session.uid, session.pid)
        tr_lam = lam ** 2
        return -tr_lam * session.pr_delta

    def _update_session_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        # lam, lam_user_d, lam_projects_d = lambdas_by_project[session.uid][session.pid]
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(session.uid, session.pid)
        # lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
        lam2 = lam ** 2
        tau = session.pr_delta
        exp_plus = np.exp(-lam2 * (tau + self.eps))
        exp_minus = np.exp(-lam2 * max(0, tau - self.eps))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                        -exp_plus + exp_minus)
            if w and w[0].category == RuntimeWarning:
                cur_ll_d = 0
                warnings.warn("in derivative", RuntimeWarning)
        # cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
        #         -exp_plus + exp_minus)

        if math.isnan(cur_ll_d):
            cur_ll_d = 0
        users_derivatives[session.uid] += cur_ll_d * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id]

    def _update_last_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        # lam, lam_user_d, lam_projects_d = lambdas_by_project.get[session.uid][session.pid]
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(session.uid, session.pid)
        # lam, lam_user_d, lam_projects_d = lambdas_by_project[user_session.pid]
        coeff = 2 * lam * session.pr_delta
        users_derivatives[session.uid] -= coeff * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] -= coeff * lam_projects_d[project_id]


class ModelExpLambda(Model):
    def __init__(self, users_histories, dim, learning_rate=0.01, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, users_histories, dim, learning_rate, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, square = False)
        self.square = False

    def _session_likelihood(self, session, lambdas_by_project):
        lam = lambdas_by_project[session.uid][session.pid]
        trans_lambda = 0.001 * np.exp(lam)
        ans = np.log(-np.exp(-trans_lambda * (session.pr_delta + self.eps)) +
                     np.exp(-trans_lambda * max(0.1, session.pr_delta - self.eps)))
        return ans

    def _last_likelihood(self, session, lambdas_by_project):
        lam = lambdas_by_project[session.uid][session.pid]
        tr_lam = 0.001 * np.exp(lam)
        return -(tr_lam) * session.pr_delta

    def _update_session_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project[session.uid][session.pid]
        tr_lam = 0.001 * np.exp(lam)
        tau = session.pr_delta
        exp_plus = np.exp(-tr_lam * (tau + self.eps))
        exp_minus = np.exp(-tr_lam * max(0, tau - self.eps))

        cur_ll_d = tr_lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                -exp_plus + exp_minus)

        if math.isnan(cur_ll_d):
            cur_ll_d = 0
        users_derivatives[session.uid] += cur_ll_d * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id]

    def _update_last_derivative(self, session, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get[session.uid][session.pid]
        coeff = 0.001 * np.exp(lam) * session.pr_delta
        users_derivatives[session.uid] -= coeff * lam_user_d
        for project_id in lam_projects_d:
            project_derivatives[project_id] -= coeff * lam_projects_d[project_id]


class ApplicableModel:
    def __init__(self, user_embeddings, project_embeddings, beta, other_project_importance, default_lambda,
                 lambda_transform=lambda x: x ** 2):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings

        mean_user = np.zeros_like(self.user_embeddings[0])
        mean_project = np.zeros_like(self.project_embeddings[0])
        for user_embedding in self.user_embeddings.values():
            mean_user += user_embedding
        for project_embedding in self.project_embeddings.values():
            mean_project += project_embedding
        self.user_embeddings[-1] = mean_user
        self.project_embeddings[-1] = mean_project

        self.beta = beta
        self.lambda_transform = lambda_transform
        self.other_project_importance = other_project_importance
        self.interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        self.lambdas_by_project = UserProjectLambdaManagerLookAhead(
            self.user_embeddings, self.project_embeddings, self.interaction_calculator, self.beta,
            self.other_project_importance, default_lambda=default_lambda, lambda_confidence=1, derivative=False,
            accum=True, square=False)

    def accept(self, session):
        self.lambdas_by_project.accept(session)

    def get_lambda(self, user_id, project_id):
        return self.lambda_transform(self.lambdas_by_project.get(user_id, project_id))

    def time_delta(self, user_id, project_id, size=1):
        lam = self.get_lambda(user_id, project_id)
        return np.mean(np.random.exponential(scale=1 / lam, size=size))

    def fit(self, events):
        for session in events:
            self.accept(session)
        return self
