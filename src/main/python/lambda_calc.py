import numpy as np

DEFAULT_FOREIGN_COEFFICIENT = .3


class InteractionCalculator:
    def __init__(self, user_embeddings, project_embeddings):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings

    def get_interaction(self, user_id, project_id):
        raise NotImplementedError()

    def get_user_supplier(self, user_id):
        raise NotImplementedError()


class SimpleInteractionsCalculator(InteractionCalculator):
    def __init__(self, user_embeddings, project_embeddings):
        super().__init__(user_embeddings, project_embeddings)

    def get_interaction(self, user_id, project_id):
        return self.user_embeddings[user_id] @ self.project_embeddings[project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.user_embeddings[user_id] @ self.project_embeddings[project_id]


class MatrixInteractionCalculator(InteractionCalculator):
    def __init__(self, user_embeddings, project_embeddings):
        super().__init__(user_embeddings, project_embeddings)
        self.interactions = user_embeddings @ project_embeddings.T

    def get_interaction(self, user_id, project_id):
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.interactions[user_id, project_id]


class LazyInteractionsCalculator(InteractionCalculator):
    def __init__(self, user_embeddings, project_embeddings):
        super().__init__(user_embeddings, project_embeddings)
        self.interactions = {}

    def get_interaction(self, user_id, project_id):
        if (user_id, project_id) not in self.interactions:
            self.interactions[user_id, project_id] = self.user_embeddings[user_id] @ \
                                                     self.project_embeddings[project_id].T
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.get_interaction(user_id, project_id)


class RowInteractionsCalculator(InteractionCalculator):
    def __init__(self, user_embeddings, project_embeddings, projects_indexes, reversed_indexes):
        super().__init__(user_embeddings, project_embeddings)
        self.projects_indexes = projects_indexes
        self.reversed_indexes = reversed_indexes
        self.interactions = {user_id: (self.user_embeddings[user_id]
                                       @ self.project_embeddings[self.projects_indexes[user_id]].T)
                             for user_id in self.user_embeddings}

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


class LazyRowInteractionsCalculator(InteractionCalculator):
    def __init__(self, user_embeddings, project_embeddings, projects_indexes, reversed_indexes):
        super().__init__(user_embeddings, project_embeddings)
        self.projects_indexes = projects_indexes
        self.reversed_indexes = reversed_indexes
        self.interactions = {}

    def _ensure_calculated(self, user_id):
        if user_id not in self.interactions:
            self.interactions[user_id] = self.user_embeddings[user_id] @ \
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
                 default_lambda=0.1, lambda_confidence=10, derivative=False, square=False):
        self.dim = len(user_embedding)
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.interactions_supplier = interactions_supplier
        self.avg_time_between_sessions = 1
        self.numerator = default_lambda * lambda_confidence
        self.denominator = lambda_confidence
        self.user_d_numerator = np.zeros_like(user_embedding)
        self.num_denom_by_project = np.zeros((1, 2))
        self.user_d_numerators_by_project = np.zeros((1, self.dim))
        self.project_d_numerator_by_project = np.zeros((1, self.dim))
        self.projects_to_ind = {}

        self.last_user_derivative_numerator = np.zeros_like(user_embedding)
        self.last_project_derivative_numerator = {}

    def update(self, project_embedding, session, delta_time):
        # delta_time is None when user get first project
        e = np.exp(-self.beta) if delta_time is not None and delta_time >= 0 else 1.  # * delta_time)
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


class UserProjectLambdaManager:
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                 lambda_confidence, derivative, accum=True, square=False):
        self.user_lambdas = {user_id: UserLambda(user_embeddings[user_id], beta, other_project_importance,
                                                 interaction_calculator.get_user_supplier(user_id),
                                                 default_lambda=default_lambda,
                                                 lambda_confidence=lambda_confidence,
                                                 derivative=derivative, square=square)
                             for user_id in user_embeddings.keys()}
        # lambdas_by_project = {user_id: {pid: 0 for pid in project_embeddings.keys()} for user_id in user_embeddings.keys()}
        self.prev_user_action_time = {}
        self.project_embeddings = project_embeddings
        self.accum = accum

    def get(self, user_id, project_id):
        raise NotImplementedError()

    def accept(self, user_id, session):
        raise NotImplementedError()


class UserProjectLambdaManagerLookAhead(UserProjectLambdaManager):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                 lambda_confidence, derivative, accum, square):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                         lambda_confidence, derivative, accum, square)

    def get(self, user_id, project_id):
        return self.user_lambdas[user_id].get(project_id, accum=self.accum)

    def accept(self, user_id, session):
        if user_id not in self.prev_user_action_time:
            # it's wrong, we can update even if it's first item of user
            # if default lambda = u^T \cdot i, we get different lambdas for unseen projects
            self.prev_user_action_time[user_id] = session.start_ts
        else:
            self.user_lambdas[user_id].update(self.project_embeddings[session.pid], session,
                                              session.start_ts - self.prev_user_action_time[user_id])


class UserProjectLambdaManagerNotLookAhead(UserProjectLambdaManager):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                 lambda_confidence, derivative, accum=True, square=False):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                         lambda_confidence, derivative, accum, square)
        self.saved_lambdas_by_project = {user_id: {pid: -1 for pid in project_embeddings.keys()} for user_id in
                                         user_embeddings.keys()}

    def get(self, user_id, project_id):
        assert self.saved_lambdas_by_project[user_id][project_id] != -1
        return self.saved_lambdas_by_project[user_id][project_id]

    def accept(self, user_id, session):
        if user_id not in self.prev_user_action_time:
            self.prev_user_action_time[user_id] = session.start_ts
        else:
            self.user_lambdas[user_id].update(self.project_embeddings[session.pid], session,
                                              session.start_ts - self.prev_user_action_time[user_id])
            self.saved_lambdas_by_project[user_id][session.pid] = self.user_lambdas[user_id].get(session.pid, accum=self.accum)


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
