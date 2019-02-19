import numpy as np

INVALID = "INVALID"


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


class UserLambda:
    def __init__(self, user_embedding, project_embeddings, beta, other_project_importance, interactions_supplier):
        self.user_embedding = user_embedding
        self.project_embeddings = project_embeddings
        self.beta = beta
        self.other_project_importance = other_project_importance
        self.interactions_supplier = interactions_supplier

        self.num_of_events = 0
        self.last_time_of_projects = {}

        self.common_sum = 0
        self.additional_sum_by_project = {}

        self.common_user_derivative = np.zeros_like(self.user_embedding)
        self.user_derivative_by_project = {}

        self.common_project_derivative = np.zeros_like(self.user_embedding)
        self.project_derivative_by_project = {}

    def update(self, project_id):
        if project_id not in self.last_time_of_projects:
            self.last_time_of_projects[project_id] = self.num_of_events
        self._update_lambda(project_id)
        self._update_user_derivative(project_id)
        self._update_project_derivative(project_id)
        self.last_time_of_projects[project_id] = self.num_of_events
        self.num_of_events += 1

    def _update_lambda(self, project_id):
        e = np.exp(-self.beta)
        decay = np.exp(-self.beta * (self.num_of_events - self.last_time_of_projects[project_id]))
        if project_id not in self.additional_sum_by_project:
            self.additional_sum_by_project[project_id] = 0
        effect = self.interactions_supplier(project_id)
        self.common_sum = e * self.common_sum + self.other_project_importance * effect
        self.additional_sum_by_project[project_id] = decay * self.additional_sum_by_project[project_id] + \
                                                     (1 - self.other_project_importance) * effect

    def _update_user_derivative(self, project_id):
        e = np.exp(-self.beta)
        decay = np.exp(-self.beta * (self.num_of_events - self.last_time_of_projects[project_id]))
        if project_id not in self.user_derivative_by_project:
            self.user_derivative_by_project[project_id] = np.zeros_like(self.user_embedding)
        effect = self.project_embeddings[project_id]
        self.common_user_derivative = e * self.common_user_derivative + self.other_project_importance * effect
        self.user_derivative_by_project[project_id] = decay * self.user_derivative_by_project[project_id] + \
                                                     (1 - self.other_project_importance) * effect

    def _update_project_derivative(self, project_id):
        e = np.exp(-self.beta)
        decay = np.exp(-self.beta * (self.num_of_events - self.last_time_of_projects[project_id]))
        if project_id not in self.project_derivative_by_project:
            self.project_derivative_by_project[project_id] = np.zeros_like(self.user_embedding)
        effect = self.user_embedding
        self.common_project_derivative = e * self.common_project_derivative + self.other_project_importance * effect
        self.project_derivative_by_project[project_id] = decay * self.project_derivative_by_project[project_id] + \
                                                         (1 - self.other_project_importance) * effect

    def get_lambda(self, project_id):
        if project_id not in self.additional_sum_by_project:
            return self.common_sum + self.interactions_supplier(project_id)
        return self.common_sum + self.interactions_supplier(project_id) + self.additional_sum_by_project[project_id]

    def get_lambda_user_derivative(self, project_id):
        if project_id not in self.user_derivative_by_project:
            return self.project_embeddings[project_id] + self.common_user_derivative
        decay = self.num_of_events - self.last_time_of_projects[project_id]
        return self.project_embeddings[project_id] + self.common_user_derivative + \
               decay * self.user_derivative_by_project[project_id]

    def get_lambda_project_derivative(self, project_id):
        derivative = {}
        for pid in self.last_time_of_projects:
            derivative[pid] = self.common_project_derivative
        if project_id not in self.project_derivative_by_project:
            return derivative
        decay = self.num_of_events - self.last_time_of_projects[project_id]
        derivative[project_id] += self.user_embedding + decay * self.project_derivative_by_project[project_id]
        return derivative
        # decay = self.num_of_events - self.last_time_of_projects[project_id]
        # return self.user_embedding + decay * self.project_weight[project_id]


# class UserLambda:
#     def __init__(self, user_embedding, beta, other_project_importance, interactions_supplier,
#                  default_lambda=0.1, lambda_confidence=10, derivative=False):
#         self.dim = len(user_embedding)
#         self.beta = beta
#         self.other_project_importance = other_project_importance
#         self.derivative = derivative
#         self.user_embedding = user_embedding
#         self.interactions_supplier = interactions_supplier
#         self.avg_time_between_events = 1
#         self.event_sum = default_lambda * lambda_confidence
#         self.user_d_event_sums = np.zeros_like(user_embedding)
#         self.event_sums_by_project = np.zeros(1)
#         self.user_d_event_sums_by_project = np.zeros((1, self.dim))
#         self.project_d_event_sums_by_project = np.zeros((1, self.dim))
#         self.projects_to_ind = {}
#
#         self.last_user_derivative_numerator = np.zeros_like(user_embedding)
#         self.last_project_derivative_numerator = {}
#
#     def update(self, project_embedding, event, delta_time):
#         # delta_time is None when user get first project
#         e = np.exp(-self.beta) if delta_time is not None and delta_time >= 0 else 1.  # * delta_time)
#         ua = self.interactions_supplier(event.pid)
#         if event.pid not in self.projects_to_ind:
#             self.projects_to_ind[event.pid] = len(self.projects_to_ind)
#             if len(self.projects_to_ind) != 1:
#                 self.event_sums_by_project = np.concatenate((self.event_sums_by_project, np.zeros(1)))
#                 self.user_d_event_sums_by_project = np.vstack(
#                     (self.user_d_event_sums_by_project, np.zeros((1, self.dim))))
#                 self.project_d_event_sums_by_project = np.vstack(
#                     (self.project_d_event_sums_by_project, np.zeros((1, self.dim))))
#
#         self.event_sum = e * self.event_sum + self.other_project_importance * ua * event.n_tasks
#         self.user_d_event_sums = e * self.user_d_event_sums + self.other_project_importance * project_embedding * event.n_tasks
#
#         self.event_sums_by_project *= e
#         self.user_d_event_sums_by_project *= e
#         self.project_d_event_sums_by_project *= e
#
#         self.event_sums_by_project[self.projects_to_ind[event.pid]] += \
#             (1 - self.other_project_importance) * ua * event.n_tasks
#         self.user_d_event_sums_by_project[self.projects_to_ind[event.pid]] += \
#             (1 - self.other_project_importance) * project_embedding * event.n_tasks
#         self.project_d_event_sums_by_project[self.projects_to_ind[event.pid]] += \
#             (1 - self.other_project_importance) * self.user_embedding * event.n_tasks
#
#         self.last_user_derivative_numerator = project_embedding * event.n_tasks
#         self.last_project_derivative_numerator = {event.pid: self.user_embedding * event.n_tasks}
#
#     def get(self, project_id, accum=True):
#         if self.derivative:
#             if accum:
#                 return self._get_accum_derivatives(project_id)
#             else:
#                 return self._get_elem_derivatives(project_id)
#         else:
#             return self._get_lambda(project_id)
#
#     def _get_lambda(self, project_id):
#         if project_id not in self.projects_to_ind:
#             return self.event_sum / self.avg_time_between_events
#         else:
#             return (self.event_sum + self.event_sums_by_project[self.projects_to_ind[project_id]]) / \
#                    self.avg_time_between_events
#
#     def _get_accum_derivatives(self, project_id):
#         if project_id not in self.projects_to_ind:
#             return self.event_sum / self.avg_time_between_events, np.zeros_like(self.user_embedding), {}
#         else:
#             project_derivatives = {}
#             for pid in self.projects_to_ind:
#                 event_sum = self.project_d_event_sums_by_project[self.projects_to_ind[pid]].copy()
#                 if pid == project_id:
#                     event_sum += self.project_d_event_sums_by_project[self.projects_to_ind[project_id]] / (
#                             1 - self.other_project_importance)
#                 project_derivatives[pid] = event_sum / self.avg_time_between_events
#
#             return (self.event_sum + self.event_sums_by_project[self.projects_to_ind[project_id]]) / \
#                    self.avg_time_between_events, \
#                    (self.user_d_event_sums + self.user_d_event_sums_by_project[self.projects_to_ind[project_id]]) / \
#                    self.avg_time_between_events, project_derivatives
#
#     def _get_elem_derivatives(self, project_id):
#         if project_id not in self.projects_to_ind:
#             return self.event_sum / self.avg_time_between_events, \
#                    np.zeros_like(self.user_embedding), {}
#         else:
#             project_derivatives = {}
#             if project_id in self.last_project_derivative_numerator:
#                 project_derivatives[project_id] = \
#                     self.last_project_derivative_numerator[project_id].copy() / self.avg_time_between_events
#
#             return (self.event_sum + self.event_sums_by_project[self.projects_to_ind[project_id]]) / \
#                    self.avg_time_between_events, \
#                    self.last_user_derivative_numerator / self.avg_time_between_events, project_derivatives
