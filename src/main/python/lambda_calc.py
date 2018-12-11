import numpy as np

# from src.main.python import wheel


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


class UserLambda:
    def __init__(self, user_embedding, beta, other_project_importance, interactions_supplier,
                 derivative=False, square=False):
        self.dim = len(user_embedding)
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.interactions_supplier = interactions_supplier
        self.avg_time_between_sessions = 5
        self.numerator = 10.
        self.denominator = 10.
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.num_denom_by_project = np.zeros((1, 2))
        self.user_d_numerators_by_project = np.zeros((1, self.dim))
        self.project_d_numerator_by_project = np.zeros((1, self.dim))
        self.projects_to_ind = {}

        self.last_user_derivative_numerator = np.zeros_like(user_embedding)
        self.last_project_derivative_numerator = {}

    def update(self, project_embedding, session, delta_time):
        e = np.exp(-self.beta)  # * delta_time)
        ua = self.interactions_supplier(session.pid)
        if session.pid not in self.projects_to_ind:
            self.projects_to_ind[session.pid] = len(self.projects_to_ind)
            if len(self.projects_to_ind) != 1:
                self.num_denom_by_project = np.vstack((self.num_denom_by_project, np.zeros((1, 2))))
                self.user_d_numerators_by_project = np.vstack((self.user_d_numerators_by_project, np.zeros((1, self.dim))))
                self.project_d_numerator_by_project = np.vstack((self.project_d_numerator_by_project, np.zeros((1, self.dim))))

        self.numerator = e * self.numerator + self.other_project_importance * ua * session.n_tasks
        self.denominator = e * self.denominator + self.other_project_importance
        self.user_d_numerator = e * self.user_d_numerator + self.other_project_importance * project_embedding * session.n_tasks

        self.num_denom_by_project *= e
        self.user_d_numerators_by_project *= e
        self.project_d_numerator_by_project *= e

        self.num_denom_by_project[self.projects_to_ind[session.pid]][0] += (1 - self.other_project_importance) * ua * session.n_tasks
        self.num_denom_by_project[self.projects_to_ind[session.pid]][1] += (1 - self.other_project_importance)
        self.user_d_numerators_by_project[self.projects_to_ind[session.pid]] += (1 - self.other_project_importance) * project_embedding * session.n_tasks
        self.project_d_numerator_by_project[self.projects_to_ind[session.pid]] += (1 - self.other_project_importance) * self.user_embedding * session.n_tasks

        self.last_user_derivative_numerator = project_embedding * session.n_tasks
        self.last_project_derivative_numerator = {session.pid: self.user_embedding * session.n_tasks}

    def get(self, project_id, accum=True):
        if self.derivative:
            if accum:
                return self._get_accum_derivatives(project_id)
            else:
                return self._get_elem_derivatives(project_id)
        else:
            return self._get_lambda(project_id)

    def _get_lambda(self, project_id):
        if project_id not in self.projects_to_ind:
            return self.numerator / self.denominator / self.avg_time_between_sessions
        else:
            return (self.numerator + self.num_denom_by_project[self.projects_to_ind[project_id]][0]) / (
                    self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]) / self.avg_time_between_sessions

    def _get_accum_derivatives(self, project_id):
        if project_id not in self.projects_to_ind:
            return self.numerator / self.denominator / self.avg_time_between_sessions, \
                   np.zeros_like(self.user_embedding), {}
        else:
            project_derivatives = {}
            denominator = self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]
            for pid in self.projects_to_ind:
                numerator = self.project_d_numerator_by_project[self.projects_to_ind[pid]].copy()
                if pid == project_id:
                    numerator += self.project_d_numerator_by_project[self.projects_to_ind[project_id]] / (
                            1 - self.other_project_importance)
                project_derivatives[pid] = numerator / denominator / self.avg_time_between_sessions

            return (self.numerator + self.num_denom_by_project[self.projects_to_ind[project_id]][0]) / (
                    self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]) / self.avg_time_between_sessions, \
                   (self.user_d_numerator + self.user_d_numerators_by_project[self.projects_to_ind[project_id]]) / (
                    self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]) / self.avg_time_between_sessions, \
                   project_derivatives

    def _get_elem_derivatives(self, project_id):
        if project_id not in self.projects_to_ind:
            return self.numerator / self.denominator / self.avg_time_between_sessions, \
                   np.zeros_like(self.user_embedding), {}
        else:
            project_derivatives = {}
            denominator = self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]
            if project_id in self.last_project_derivative_numerator:
                project_derivatives[project_id] = self.last_project_derivative_numerator[project_id].copy() \
                                                  / denominator / self.avg_time_between_sessions

            return (self.numerator + self.num_denom_by_project[self.projects_to_ind[project_id]][0]) / (
                    self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]) / self.avg_time_between_sessions, \
                   self.last_user_derivative_numerator / (
                           self.denominator + self.num_denom_by_project[self.projects_to_ind[project_id]][1]) / self.avg_time_between_sessions, \
                   project_derivatives


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
    # wheel.calc_lambdas(project_id, user_embedding, projects_embeddings, dim, beta, interactions, derivative,
    #                    DEFAULT_FOREIGN_COEFFICIENT, project_ids, n_tasks, time_deltas, out_lambdas,
    #                    out_user_derivatives, out_project_derivatives)
    if derivative:
        return out_lambdas, out_user_derivatives, out_project_derivatives
    return out_lambdas
