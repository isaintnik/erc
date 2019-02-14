import math
import warnings

from collections import namedtuple
import numpy as np

from src.main.python.lambda_calc import UserLambda, LazyInteractionsCalculator, SimpleInteractionsCalculator, \
    UserProjectLambdaManagerLookAhead, UserProjectLambdaManagerNotLookAhead, INVALID
from src.test.python.metrics import print_metrics


Event = namedtuple('Event', 'uid pid start_ts pr_delta n_tasks')


class Model:
    def __init__(self, dim, beta=0.001, eps=1, other_project_importance=0.1,
                 users_embeddings_prior=None, projects_embeddings_prior=None, lambda_transform=lambda x: x ** 2,
                 lambda_derivative=lambda x: 2 * x):
        self.emb_dim = dim
        self.decay_rate = 1  # 0.95
        self.beta = beta
        self.eps = eps
        self.other_project_importance = other_project_importance
        self.lambda_transform = lambda_transform
        self.lambda_derivative = lambda_derivative
        self.user_embeddings = users_embeddings_prior if users_embeddings_prior else users_embeddings_prior
        self.project_embeddings = projects_embeddings_prior if users_embeddings_prior else None
        self.data_inited = False

    def log_likelihood(self, events):
        return self._likelihood_derivative(events)

    def ll_derivative(self, events):
        return self._likelihood_derivative(events, derivative=True)

    def fit(self, data, learning_rate, iter_num, optim_type="sgd", eval=None, verbose=True):
        if optim_type == "sgd":
            self._sgd_optimization(data, learning_rate, iter_num, eval, verbose)
        else:
            self._glove_like_optimisation(data, learning_rate, iter_num, eval, verbose)

    def get_applicable(self, data):
        return ApplicableModel(self.user_embeddings, self.project_embeddings, self.beta,
                               self.other_project_importance, self.default_lambda,
                               lambda_transform=self.lambda_transform).fit(data)

    def _init_data(self, events):
        self.data_size = len(events)
        self.user_ids = set(event.uid for event in events)
        self.project_ids = set(event.pid for event in events)
        self.user_embeddings = {user_id: np.abs(np.random.normal(0.1, 0.1, self.emb_dim))
                                for user_id in self.user_ids} \
            if self.user_embeddings is None else self.user_embeddings
        if self.project_embeddings is None:
            self.project_embeddings = {pid: np.abs(np.random.normal(0.1, 0.1, self.emb_dim)) for pid in self.project_ids}

        self._update_default_embeddings()

        self.default_lambda = 1.
        self.lambda_confidence = 1.
        pr_deltas = []
        for event in events:
            if event.pr_delta is not None and not math.isnan(event.pr_delta):
                pr_deltas.append(event.pr_delta)
        pr_deltas = np.array(pr_deltas)
        self.default_lambda = 1 / np.mean(pr_deltas)
        # print("default_lambda", self.default_lambda)

    def _likelihood_derivative(self, events, derivative=False):
        if not self.data_inited:
            self._init_data(events)
        ll = 0.
        users_derivatives = {user_id: np.zeros_like(self.user_embeddings[user_id]) for user_id in self.user_ids}
        project_derivatives = {project_id: np.zeros_like(self.project_embeddings[project_id]) for project_id in self.project_ids}
        done_projects = {user_id: set() for user_id in self.user_ids}
        last_times_events = set()
        interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        lambdas_by_project = UserProjectLambdaManagerNotLookAhead(
            self.user_embeddings, self.project_embeddings, interaction_calculator, self.beta,
            self.other_project_importance, self.default_lambda, self.lambda_confidence, derivative, accum=True)
        for event in events:
            # first time done projects tasks, skip ll update
            # i forget why we do like this
            if event.pid not in done_projects[event.uid]:
                done_projects[event.uid].add(event.pid)
                lambdas_by_project.accept(event)
                continue

            if event.n_tasks != 0:
                if derivative:
                    self._update_event_derivative(event, lambdas_by_project, users_derivatives,
                                                  project_derivatives)
                else:
                    ll += self._event_likelihood(event, lambdas_by_project)
                lambdas_by_project.accept(event)
            else:
                last_times_events.add(event)

        for event in last_times_events:
            if derivative:
                self._update_last_derivative(event, lambdas_by_project, users_derivatives,
                                             project_derivatives)
            else:
                ll += self._last_likelihood(event, lambdas_by_project)
        if derivative:
            return users_derivatives, project_derivatives
        return ll

    def _glove_like_optimisation(self, data, learning_rate, iter_num=30, eval=None, verbose=False):
        if not self.data_inited:
            self._init_data(data)
        discount_decay = 0.99
        users_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.user_embeddings.items()}
        projects_diffs_squares = {k: np.ones_like(v) * 1 for k, v in self.project_embeddings.items()}
        interaction_calculator = SimpleInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        done_projects = {user_id: set() for user_id in self.user_embeddings}
        last_times_events = set()
        lambdas_by_project = UserProjectLambdaManagerNotLookAhead(
            self.user_embeddings, self.project_embeddings, interaction_calculator, self.beta, self.other_project_importance,
            self.default_lambda, self.lambda_confidence, derivative=True, accum=False, square=True
        )
        for optimization_iter in range(iter_num):
            for event in data:
                if event.pr_delta is None or event.pid not in done_projects[event.uid]:
                    done_projects[event.uid].add(event.pid)
                    lambdas_by_project.accept(event)
                    continue
                # assert not (event.pid in done_projects[event.uid] and event.pr_delta is None)
                if event.n_tasks != 0:
                    self._update_glove_event_params(event, lambdas_by_project, users_diffs_squares,
                                                      projects_diffs_squares, discount_decay, learning_rate)
                    lambdas_by_project.accept(event)
                else:
                    last_times_events.add(event)
            for event in last_times_events:
                self._update_glove_last_params(event, lambdas_by_project, users_diffs_squares,
                                               projects_diffs_squares, discount_decay, learning_rate)
            learning_rate *= self.decay_rate
            if verbose:
                print("{}-th iter, ll = {}".format(optimization_iter, self.log_likelihood(data)))
                if eval is not None:
                    print_metrics(self, train_data=data, test_data=eval, samples_num=10)
                print()

    def _update_glove_event_params(self, event, lambdas_by_project, users_diffs_squares,
                                   projects_diffs_squares, discount_decay, lr):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(event.uid, event.pid)
        # lam, lam_user_d, lam_projects_d = user_lambda.get(user_event.pid, accum=False)
        tr_lam = lam ** 2
        tau = event.pr_delta
        exp_plus = np.exp(-tr_lam * (tau + self.eps))
        exp_minus = np.exp(-tr_lam * max(0, tau - self.eps))
        cur_ll_d = 2 * lam * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                -exp_plus + exp_minus)
        if math.isnan(cur_ll_d):
            cur_ll_d = 0.

        user_diff = cur_ll_d * lam_user_d
        users_diffs_squares[event.uid] = discount_decay * users_diffs_squares[event.uid] + user_diff * user_diff
        self.user_embeddings[event.uid] += user_diff * lr / np.sqrt(users_diffs_squares[event.uid])
        for project_id in lam_projects_d:
            project_diff = cur_ll_d * lam_projects_d[project_id]
            projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[
                project_id] + project_diff * project_diff
            self.project_embeddings[project_id] += project_diff * lr / np.sqrt(
                projects_diffs_squares[project_id])

    def _update_glove_last_params(self, event, lambdas_by_project, users_diffs_squares,
                                  projects_diffs_squares, discount_decay, lr):
        # update in the end
        # lam, lam_user_d, lam_projects_d = user_lambda.get(user_event.pid, accum=False)
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(event.uid, event.pid)
        user_diff = 2 * lam * event.pr_delta * lam_user_d
        users_diffs_squares[event.uid] = discount_decay * users_diffs_squares[event.uid] + user_diff * user_diff
        self.user_embeddings[event.uid] -= user_diff * lr / np.sqrt(users_diffs_squares[event.uid])
        for project_id in lam_projects_d:
            project_diff = 2 * lam * event.pr_delta * lam_projects_d[project_id]
            projects_diffs_squares[project_id] = discount_decay * projects_diffs_squares[
                project_id] + project_diff * project_diff
            self.project_embeddings[project_id] -= project_diff * lr / np.sqrt(projects_diffs_squares[project_id])

    def _event_likelihood(self, event, lambdas_by_project):
        lam = lambdas_by_project.get(event.uid, event.pid)
        tr_lam = self.lambda_transform(lam)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = np.log(-np.exp(-tr_lam * (event.pr_delta + self.eps)) +
                         np.exp(-tr_lam * max(0, event.pr_delta - self.eps)))
            if w and w[0].category == RuntimeWarning:
                ans = 0
                warnings.warn("in ll", RuntimeWarning)
        return ans

    def _last_likelihood(self, event, lambdas_by_project):
        lam = lambdas_by_project.get(event.uid, event.pid)
        tr_lam = self.lambda_transform(lam)
        return -tr_lam * event.pr_delta

    def _update_event_derivative(self, event, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(event.uid, event.pid)
        tr_lam = self.lambda_transform(lam)
        tau = event.pr_delta
        exp_plus = np.exp(-tr_lam * (tau + self.eps))
        exp_minus = np.exp(-tr_lam * max(0, tau - self.eps))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cur_ll_d = self.lambda_derivative(lam) * ((tau + self.eps) * exp_plus - max(0, tau - self.eps) * exp_minus) / (
                    -exp_plus + exp_minus)
            if w and w[0].category == RuntimeWarning:
                cur_ll_d = 0
                warnings.warn("in derivative", RuntimeWarning)

        if math.isnan(cur_ll_d):
            cur_ll_d = 0
        users_derivatives[event.uid] += cur_ll_d * lam_user_d
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id]

    def _update_last_derivative(self, event, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(event.uid, event.pid)
        coeff = self.lambda_derivative(lam) * event.pr_delta
        users_derivatives[event.uid] -= coeff * lam_user_d
        # so slow
        for project_id in lam_projects_d:
            project_derivatives[project_id] -= coeff * lam_projects_d[project_id]

    def _sgd_optimization(self, data, learning_rate, iter_num, eval, verbose=True):
        if not self.data_inited:
            self._init_data(data)
        for i in range(iter_num):
            users_derivatives, project_derivatives = self.ll_derivative(data)
            lr = learning_rate / self.data_size
            for user_id in self.user_ids:
                self.user_embeddings[user_id] += users_derivatives[user_id] * lr
            for project_id in self.project_ids:
                self.project_embeddings[project_id] += project_derivatives[project_id] * lr
            learning_rate *= self.decay_rate
            self._update_default_embeddings()

            if verbose:
                print("{}-th iter, ll = {}".format(i, self.log_likelihood(data)))
                if eval is not None:
                    print_metrics(self, train_data=data, test_data=eval, samples_num=10)
                print()

    def _update_default_embeddings(self):
        mean_user = np.zeros_like(next(iter(self.user_embeddings.values())))
        mean_project = np.zeros_like(next(iter(self.project_embeddings.values())))
        for user_embedding in self.user_embeddings.values():
            mean_user += user_embedding
        for project_embedding in self.project_embeddings.values():
            mean_project += project_embedding
        self.user_embeddings[INVALID] = mean_user / len(self.user_embeddings)
        self.project_embeddings[INVALID] = mean_project / len(self.project_embeddings)


class Model2Lambda(Model):
    def __init__(self, dim, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, dim, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, lambda_transform=lambda x: x ** 2,
                       lambda_derivative=lambda x: 2 * x)


class ModelExpLambda(Model):
    def __init__(self, dim, beta=0.01, eps=3600, other_project_importance=0.5,
                 users_embeddings_prior=None, projects_embeddings_prior=None):
        Model.__init__(self, dim, beta, eps, other_project_importance,
                       users_embeddings_prior, projects_embeddings_prior, lambda_transform=lambda x: 1 * np.exp(x),
                       lambda_derivative=lambda x: 1 * np.exp(x))


class ApplicableModel:
    def __init__(self, user_embeddings, project_embeddings, beta, other_project_importance, default_lambda,
                 lambda_transform):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings

        mean_user = np.zeros_like(next(iter(self.user_embeddings.values())))
        mean_project = np.zeros_like(next(iter(self.project_embeddings.values())))
        for user_embedding in self.user_embeddings.values():
            mean_user += user_embedding
        for project_embedding in self.project_embeddings.values():
            mean_project += project_embedding
        self.user_embeddings[INVALID] = mean_user / len(self.user_embeddings)
        self.project_embeddings[INVALID] = mean_project / len(self.project_embeddings)

        self.beta = beta
        self.lambda_transform = lambda_transform
        self.other_project_importance = other_project_importance
        self.interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        self.lambdas_by_project = UserProjectLambdaManagerLookAhead(
            self.user_embeddings, self.project_embeddings, self.interaction_calculator, self.beta,
            self.other_project_importance, default_lambda=default_lambda, lambda_confidence=1, derivative=False,
            accum=True)

    def accept(self, event):
        self.lambdas_by_project.accept(event)

    def get_lambda(self, user_id, project_id):
        return self.lambda_transform(self.lambdas_by_project.get(user_id, project_id))

    def time_delta(self, user_id, project_id, size=1):
        lam = self.get_lambda(user_id, project_id)
        # return np.mean(np.random.exponential(scale=1 / lam, size=size))
        return 1 / lam

    def fit(self, events):
        for event in events:
            self.accept(event)
        return self


class ModelDensity:
    def __init__(self, dim, beta=0.001, eps=1, other_project_importance=0.3,
                 users_embeddings_prior=None, projects_embeddings_prior=None, lambda_transform=lambda x: x ** 2,
                 lambda_derivative=lambda x: 2 * x):
        self.emb_dim = dim
        self.decay_rate = 1  # 0.95
        self.beta = beta
        self.eps = eps
        self.other_project_importance = other_project_importance
        self.lambda_transform = lambda_transform
        self.lambda_derivative = lambda_derivative
        self.user_embeddings = users_embeddings_prior if users_embeddings_prior else users_embeddings_prior
        self.project_embeddings = projects_embeddings_prior if users_embeddings_prior else None
        self.data_inited = False

    def _init_data(self, events):
        self.data_size = len(events)
        self.user_ids = set(event.uid for event in events)
        self.project_ids = set(event.pid for event in events)
        self.user_embeddings = {user_id: np.abs(np.random.normal(0.1, 0.1, self.emb_dim))
                                for user_id in self.user_ids} \
            if self.user_embeddings is None else self.user_embeddings
        if self.project_embeddings is None:
            self.project_embeddings = {pid: np.abs(np.random.normal(0.1, 0.1, self.emb_dim)) for pid in self.project_ids}

        self._update_default_embeddings()

        self.default_lambda = 1.
        self.lambda_confidence = 1.
        pr_deltas = []
        for event in events:
            if event.pr_delta is not None and not math.isnan(event.pr_delta):
                pr_deltas.append(event.pr_delta)
        pr_deltas = np.array(pr_deltas)
        self.default_lambda = 1 / np.mean(pr_deltas)
        # print("default_lambda", self.default_lambda)

    def log_likelihood(self, data):
        return self._likelihood_derivative(data)

    def ll_derivative(self, data):
        return self._likelihood_derivative(data, derivative=True)

    def _likelihood_derivative(self, data, derivative=False):
        if not self.data_inited:
            self._init_data(data)
        ll = 0.
        users_derivatives = {user_id: np.zeros_like(self.user_embeddings[user_id]) for user_id in self.user_ids}
        project_derivatives = {project_id: np.zeros_like(self.project_embeddings[project_id]) for project_id in self.project_ids}
        done_projects = {user_id: set() for user_id in self.user_ids}
        last_times_events = set()
        interaction_calculator = LazyInteractionsCalculator(self.user_embeddings, self.project_embeddings)
        lambdas_by_project = UserProjectLambdaManagerLookAhead(
            self.user_embeddings, self.project_embeddings, interaction_calculator, self.beta,
            self.other_project_importance, self.default_lambda, self.lambda_confidence, derivative, accum=True)
        for event in data:
            if event.pid not in done_projects[event.uid]:
                done_projects[event.uid].add(event.pid)
                lambdas_by_project.accept(event)
                continue

            if event.n_tasks != 0:
                if derivative:
                    self._update_event_derivative(event, lambdas_by_project, users_derivatives,
                                                    project_derivatives)
                else:
                    ll += self._event_likelihood(event, lambdas_by_project)
                lambdas_by_project.accept(event)
            else:
                last_times_events.add(event)

        if derivative:
            return users_derivatives, project_derivatives
        return ll

    def _event_likelihood(self, event, lambdas_by_project):
        lam = lambdas_by_project.get(event.uid, event.pid)
        tr_lam = self.lambda_transform(lam)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ans = np.log(tr_lam * np.exp(-tr_lam * (event.pr_delta)))
            if w and w[0].category == RuntimeWarning:
                ans = 0
                warnings.warn("in ll", RuntimeWarning)
        return ans

    def _update_event_derivative(self, event, lambdas_by_project, users_derivatives, project_derivatives):
        lam, lam_user_d, lam_projects_d = lambdas_by_project.get(event.uid, event.pid)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tr_lam = self.lambda_transform(lam)
            der_lam = self.lambda_derivative(lam)
            cur_ll_d = der_lam * np.exp(-tr_lam * event.pr_delta) * (1 - tr_lam * event.pr_delta) / tr_lam
            if w and w[0].category == RuntimeWarning:
                cur_ll_d = 0
                warnings.warn("in derivative", RuntimeWarning)

        if math.isnan(cur_ll_d):
            cur_ll_d = 0
        users_derivatives[event.uid] += cur_ll_d * lam_user_d
        for project_id in lam_projects_d:
            project_derivatives[project_id] += cur_ll_d * lam_projects_d[project_id]

    def get_applicable(self, data):
        return ApplicableModel(self.user_embeddings, self.project_embeddings, self.beta, self.other_project_importance,
                               self.default_lambda, lambda_transform=self.lambda_transform).fit(data)

    def _sgd_optimization(self, data, learning_rate, iter_num, eval, verbose=True):
        if not self.data_inited:
            self._init_data(data)
        for i in range(iter_num):
            users_derivatives, project_derivatives = self.ll_derivative(data)
            lr = learning_rate / self.data_size
            for user_id in self.user_ids:
                self.user_embeddings[user_id] += users_derivatives[user_id] * lr
            for project_id in self.project_ids:
                self.project_embeddings[project_id] += project_derivatives[project_id] * lr
            learning_rate *= self.decay_rate
            self._update_default_embeddings()

            if verbose:
                print("{}-th iter, ll = {}".format(i, self.log_likelihood(data)))
                if eval is not None:
                    print_metrics(self, train_data=data, test_data=eval, samples_num=10)
                print()

    def fit(self, data, learning_rate, iter_num, eval=None, verbose=True):
        self._sgd_optimization(data, learning_rate, iter_num, eval, verbose)

    def _update_default_embeddings(self):
        mean_user = np.zeros_like(next(iter(self.user_embeddings.values())))
        mean_project = np.zeros_like(next(iter(self.project_embeddings.values())))
        for user_embedding in self.user_embeddings.values():
            mean_user += user_embedding
        for project_embedding in self.project_embeddings.values():
            mean_project += project_embedding
        self.user_embeddings[INVALID] = mean_user / len(self.user_embeddings)
        self.project_embeddings[INVALID] = mean_project / len(self.project_embeddings)
