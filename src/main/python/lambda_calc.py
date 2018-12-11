import copy
import numpy as np
from src.main.python import wheel


DEFAULT_FOREIGN_COEFFICIENT = .3


# matrix multiplication or dict with pairs
class InteractionCalculator:
    def __init__(self, user_embeddings, project_embeddings, calc_type="matrix"):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.calc_type = calc_type  # matrix, lazy_dict, recalc (not cached)
        if self.calc_type == "matrix":
            self.interactions = user_embeddings @ project_embeddings.T
        else:
            self.interactions = {}

    def get_interaction(self, user_id, project_id):
        if self.calc_type == "recalc":
            return self.user_embeddings[user_id] @ self.project_embeddings[project_id].T
        elif self.calc_type == "lazy_dict":
            if (user_id, project_id) not in self.interactions:
                self.interactions[user_id, project_id] = self.user_embeddings[user_id] @ \
                                                     self.project_embeddings[project_id].T
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.get_interaction(user_id, project_id)


class SimpleInteractionsCalculator:
    def __init__(self, user_embeddings, project_embeddings, calc_type="matrix"):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings

    def get_interaction(self, user_id, project_id):
        return self.user_embeddings[user_id] @ self.project_embeddings[project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.user_embeddings[user_id] @ self.project_embeddings[project_id]


class MatrixInteractionCalculator:
    def __init(self, user_embeddings, project_embeddings):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.interactions = user_embeddings @ project_embeddings.T

    def get_interaction(self, user_id, project_id):
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.interactions[user_id, project_id]


class LazyInteractionsCalculator():
    def __init__(self, user_embeddings, project_embeddings):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.interactions = {}

    def get_interaction(self, user_id, project_id):
        if (user_id, project_id) not in self.interactions:
            self.interactions[user_id, project_id] = self.user_embeddings[user_id] @ \
                                                     self.project_embeddings[project_id].T
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.get_interaction(user_id, project_id)


class RowInteractionsCalculator:
    def __init__(self, users_embeddings, project_embeddings, projects_indexes, reversed_indexes):
        self.users_embeddings = users_embeddings
        self.project_embeddings = project_embeddings
        self.projects_indexes = projects_indexes
        self.reversed_indexes = reversed_indexes
        self.interactions = [user_embedding @ project_embeddings[user_projects].T
                             for user_embedding, user_projects in zip(users_embeddings, projects_indexes)]

    def get_interaction(self, user_id, project_id):
        return self.interactions[user_id][self.reversed_indexes[user_id][project_id]]

    def get_interaction_compressed(self, user_id, project_id_compressed):
        return self.interactions[user_id][project_id_compressed]

    def get_user_supplier(self, user_id):
        user_row = self.interactions[user_id]
        return lambda project_id: user_row[self.reversed_indexes[user_id][project_id]]

    def get_user_supplier_compressed(self, user_id):
        user_row = self.interactions[user_id]
        return lambda project_id_compressed: user_row[project_id_compressed]


class LazyRowInteractionsCalculator:
    def __init__(self, users_embeddings, project_embeddings, projects_indexes, reversed_indexes):
        self.users_embeddings = users_embeddings
        self.project_embeddings = project_embeddings
        self.projects_indexes = projects_indexes
        self.reversed_indexes = reversed_indexes
        self.interactions = {}

    def _ensure_calculated(self, user_id):
        if user_id not in self.interactions:
            self.interactions[user_id] = self.users_embeddings[user_id] @ \
                                         self.project_embeddings[self.projects_indexes[user_id]].T

    def get_interaction(self, user_id, project_id):
        self._ensure_calculated(user_id)
        return self.interactions[user_id][self.reversed_indexes[user_id][project_id]]

    def get_interaction_compressed(self, user_id, project_id_compressed):
        self._ensure_calculated(user_id)
        return self.interactions[user_id][project_id_compressed]

    def get_user_supplier(self, user_id):
        self._ensure_calculated(user_id)
        user_row = self.interactions[user_id]
        return lambda project_id: user_row[self.reversed_indexes[user_id][project_id]]

    def get_user_supplier_compressed(self, user_id):
        self._ensure_calculated(user_id)
        user_row = self.interactions[user_id]
        return lambda project_id_compressed: user_row[project_id_compressed]


class UserProjectLambda:
    def __init__(self, user_embedding, beta, interactions_supplier, *, derivative=False, square=False):
        self.numerator = 10
        self.denominator = 10
        self.beta = beta
        # self.square = square
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.project_d_numerators = {}
        self.avg_time_between_sessions = 5
        self.interactions_supplier = interactions_supplier
        self.this_project_id = None

    def set_project_id(self, project_id):
        self.this_project_id = project_id

    def update(self, project_embedding, session_project_id, n_tasks, delta_time, coeff):
        e = np.exp(-self.beta)  # * delta_time)
        ua = self.interactions_supplier(session_project_id)
        self.numerator = e * self.numerator + coeff * ua * n_tasks  # * (ua if self.square else 1)
        self.denominator = e * self.denominator + coeff
        if self.derivative:
            self.user_d_numerator = e * self.user_d_numerator + coeff * project_embedding * n_tasks  # * (2 * ua if self.square else 1)
            if self.this_project_id is not None:
                for project_id in self.project_d_numerators:
                    self.project_d_numerators[project_id] *= e
                if session_project_id not in self.project_d_numerators:
                    self.project_d_numerators[session_project_id] = np.zeros_like(self.user_embedding)
                self.project_d_numerators[self.this_project_id] += coeff * self.user_embedding * n_tasks / self.avg_time_between_sessions  # * (2 * ua if self.square else 1)

    def get(self):
        cur_lambda = self.numerator / self.denominator / self.avg_time_between_sessions
        if self.derivative:
            user_derivative = self.user_d_numerator / self.denominator / self.avg_time_between_sessions
            project_derivative = self.project_d_numerators  # / self.denominator / self.avg_time_between_sessions
            return cur_lambda, user_derivative, project_derivative, self.denominator
        return cur_lambda


class UserLambda:
    def __init__(self, user_embedding, beta, other_project_importance, interactions_supplier,
                 derivative=False, square=False):
        self.project_lambdas = {}
        self.other_project_importance = other_project_importance
        self.default_lambda = UserProjectLambda(user_embedding, beta, interactions_supplier,
                                                derivative=derivative, square=square)

    def copy_default_lambda(self, project_id):
        new_lam = copy.copy(self.default_lambda)
        new_lam.user_d_numerator = copy.deepcopy(self.default_lambda.user_d_numerator)
        new_lam.project_d_numerators = {}
        new_lam.this_project_id = project_id
        return new_lam

    def update(self, project_embedding, session, delta_time):
        # print("user embedding in UserLambda", self.default_lambda.user_embedding)
        if session.pid not in self.project_lambdas:
            self.project_lambdas[session.pid] = self.copy_default_lambda(session.pid)
        for project_id, project_lambda in self.project_lambdas.items():
            coefficient = 1 if project_id == session.pid else self.other_project_importance
            project_lambda.update(project_embedding, session.pid, session.n_tasks, delta_time, coefficient)
        self.default_lambda.update(project_embedding, session.pid, session.n_tasks,
                                   delta_time, self.other_project_importance)

    def get(self, project_id):
        if project_id not in self.project_lambdas.keys():
            return self.default_lambda.get()
        return self.project_lambdas[project_id].get()


def projects_index(history):
    used_projects = []
    used_projects_set = set()
    for session in history:
        if session.pid not in used_projects_set:
            used_projects.append(session.pid)
            used_projects_set.add(session.pid)
    return np.array(used_projects)


def reverse_projects_indices(indices):
    return {general_ind: local_ind for local_ind, general_ind in enumerate(indices)}


def convert_history(history, reversed_project_index):
    project_ids = []
    time_deltas = []
    n_tasks = []
    for session in history:
        project_ids.append(reversed_project_index[session.pid])
        time_deltas.append(session.pr_delta)
        n_tasks.append(session.n_tasks)
    project_ids = np.array(project_ids)
    time_deltas = np.array(time_deltas)
    n_tasks = np.array(n_tasks)
    return project_ids, time_deltas, n_tasks


def calc_lambdas(user_id, project_id, history, user_embedding, beta, interactions, projects_embeddings, dim,
                 derivative=False):
    upl = UserProjectLambda(user_embedding, beta, interactions.get_user_supplier(user_id), derivative=derivative)
    lambdas = np.zeros(len(history), dtype=np.float64)
    user_derivatives = None
    project_derivatives = None
    if derivative:
        user_derivatives = np.zeros((len(history), dim), dtype=np.float64)
        project_derivatives = np.zeros((len(history), len(projects_embeddings), dim), dtype=np.float64)
    for i, session in enumerate(history):
        upl.update(projects_embeddings[session.pid], session.pid, session.n_tasks, None,
                   1 if project_id == session.pid else DEFAULT_FOREIGN_COEFFICIENT)
        if derivative:
            lambdas[i], user_derivatives[i], _, _ = upl.get()
            # TODO: project derivatives
        else:
            lambdas[i] = upl.get()
    if derivative:
        return lambdas, user_derivatives, project_derivatives
    return lambdas


def calc_lambdas_native(project_id, project_ids, n_tasks, time_deltas, user_embedding, dim, beta, interactions,
                        projects_embeddings, derivative=False):
    out_lambdas = np.zeros(len(project_ids), dtype=np.float64)
    out_user_derivatives = np.zeros((len(project_ids), len(user_embedding)), dtype=np.float64)
    out_project_derivatives = np.zeros((len(project_ids),) + projects_embeddings.shape, dtype=np.float64)
    wheel.calc_lambdas(project_id, user_embedding, projects_embeddings, dim, beta, interactions, derivative,
                       DEFAULT_FOREIGN_COEFFICIENT, project_ids, n_tasks, time_deltas, out_lambdas,
                       out_user_derivatives, out_project_derivatives)
    if derivative:
        return out_lambdas, out_user_derivatives, out_project_derivatives
    return out_lambdas
