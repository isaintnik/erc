import numpy as np

INVALID = "INVALID"

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
                 default_lambda=0.1, lambda_confidence=10, derivative=False):
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
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance,
                 default_lambda, lambda_confidence, derivative, accum=True):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.interaction_calculator = interaction_calculator
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.default_lambda = default_lambda
        self.lambda_confidence=lambda_confidence
        self.derivative = derivative
        self.accum = accum

        self.user_lambdas = {user_id: UserLambda(user_embeddings[user_id], beta, other_project_importance,
                                                 self.interaction_calculator.get_user_supplier(user_id),
                                                 default_lambda=default_lambda,
                                                 lambda_confidence=lambda_confidence,
                                                 derivative=derivative)
                             for user_id in user_embeddings.keys()}
        self.prev_user_action_time = {}
        self.accum = accum

    def get(self, user_id, project_id):
        raise NotImplementedError()

    def accept(self, session):
        raise NotImplementedError()


class UserProjectLambdaManagerLookAhead(UserProjectLambdaManager):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                 lambda_confidence, derivative, accum=True):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance, default_lambda,
                         lambda_confidence, derivative, accum)

    def get(self, user_id, project_id):
        if project_id not in self.project_embeddings:
            return self.user_embeddings[user_id] @ self.project_embeddings[INVALID].T
        return self.user_lambdas[user_id].get(project_id, accum=self.accum)

    def accept(self, session):
        if session.uid not in self.prev_user_action_time:
            # it's wrong, we can update even if it's first item of user
            # if default lambda = u^T \cdot i, we get different lambdas for unseen projects
            self.prev_user_action_time[session.uid] = session.start_ts
        else:
            if session.pid not in self.project_embeddings:
                self.project_embeddings[session.pid] = np.copy(self.project_embeddings[INVALID])
            if session.uid not in self.user_embeddings:
                self.user_embeddings[session.uid] = np.copy(self.user_embeddings[INVALID])
                self.user_lambdas[session.uid] = UserLambda(
                    self.user_embeddings[session.uid], self.beta, self.other_project_importance,
                    self.interaction_calculator.get_user_supplier(session.uid), self.default_lambda,
                    self.lambda_confidence, self.derivative)
            self.user_lambdas[session.uid].update(self.project_embeddings[session.pid], session,
                                                  session.start_ts - self.prev_user_action_time[session.uid])


class UserProjectLambdaManagerNotLookAhead(UserProjectLambdaManager):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance,
                 default_lambda, lambda_confidence, derivative, accum=True, square=False):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance,
                         default_lambda, lambda_confidence, derivative, accum)
        self.saved_lambdas_by_project = {user_id: {pid: -1 for pid in project_embeddings.keys()} for user_id in
                                         user_embeddings.keys()}

    def get(self, user_id, project_id):
        if self.saved_lambdas_by_project[user_id][project_id] == -1:
            self.saved_lambdas_by_project[user_id][project_id] = self.user_lambdas[user_id].get(project_id,
                                                                                                accum=self.accum)
        return self.saved_lambdas_by_project[user_id][project_id]

    def accept(self, session):
        if session.uid not in self.prev_user_action_time:
            self.prev_user_action_time[session.uid] = session.start_ts
            self.saved_lambdas_by_project[session.uid][session.pid] = self.user_lambdas[session.uid].get(session.pid,
                                                                                                 accum=self.accum)
        else:
            if session.pid not in self.project_embeddings:
                self.project_embeddings[session.pid] = np.copy(self.project_embeddings[INVALID])
            self.user_lambdas[session.uid].update(self.project_embeddings[session.pid], session,
                                                  session.start_ts - self.prev_user_action_time[session.uid])
            self.saved_lambdas_by_project[session.uid][session.pid] = self.user_lambdas[session.uid].get(session.pid,
                                                                                                 accum=self.accum)
