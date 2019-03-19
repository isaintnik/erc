import numpy as np


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

        self.current_time = 0.
        self.last_time_of_projects = {}

        self.common_sum = 0
        self.additional_sum_by_project = {}

        self.common_user_derivative = np.zeros_like(self.user_embedding)
        self.user_derivative_by_project = {}

        self.common_project_derivative = np.zeros_like(self.user_embedding)
        self.project_derivative_by_project = {}

    def update(self, project_id, time_delta=1):
        time_delta = 1
        if project_id not in self.last_time_of_projects:
            self.last_time_of_projects[project_id] = self.current_time
        e = np.exp(-self.beta * time_delta)
        decay = np.exp(-self.beta * (self.current_time - self.last_time_of_projects[project_id]))
        self._update_lambda(project_id, e, decay)
        self._update_user_derivative(project_id, e, decay)
        self._update_project_derivative(project_id, e, decay)
        self.last_time_of_projects[project_id] = self.current_time
        self.current_time += time_delta

    def _update_lambda(self, project_id, e, project_decay):
        if project_id not in self.additional_sum_by_project:
            self.additional_sum_by_project[project_id] = 0
        effect = self.interactions_supplier(project_id)
        self.common_sum = e * self.common_sum + self.other_project_importance * effect
        self.additional_sum_by_project[project_id] = project_decay * self.additional_sum_by_project[project_id] + \
                                                     (1 - self.other_project_importance) * effect

    def _update_user_derivative(self, project_id, e, project_decay):
        if project_id not in self.user_derivative_by_project:
            self.user_derivative_by_project[project_id] = np.zeros_like(self.user_embedding)
        effect = self.project_embeddings[project_id]
        self.common_user_derivative = e * self.common_user_derivative + self.other_project_importance * effect
        self.user_derivative_by_project[project_id] = project_decay * self.user_derivative_by_project[project_id] + \
                                                      (1 - self.other_project_importance) * effect

    def _update_project_derivative(self, project_id, e, project_decay):
        if project_id not in self.project_derivative_by_project:
            self.project_derivative_by_project[project_id] = np.zeros_like(self.user_embedding)
        effect = self.user_embedding
        self.common_project_derivative = e * self.common_project_derivative + self.other_project_importance * effect
        self.project_derivative_by_project[project_id] = project_decay * self.project_derivative_by_project[project_id]\
                                                         + (1 - self.other_project_importance) * effect

    def get_lambda(self, project_id):
        if project_id not in self.additional_sum_by_project:
            return self.common_sum + self.interactions_supplier(project_id)
        return self.common_sum + self.interactions_supplier(project_id) + self.additional_sum_by_project[project_id]

    def get_lambda_user_derivative(self, project_id):
        if project_id not in self.user_derivative_by_project:
            return self.project_embeddings[project_id] + self.common_user_derivative
        # decay = self.current_time - self.last_time_of_projects[project_id]
        decay = np.exp(-self.beta * (self.current_time - self.last_time_of_projects[project_id]))
        return self.project_embeddings[project_id] + self.common_user_derivative + \
               decay * self.user_derivative_by_project[project_id]

    def get_lambda_project_derivative(self, project_id):
        derivative = {}
        for pid in self.last_time_of_projects:
            derivative[pid] = self.common_project_derivative
        if project_id not in self.project_derivative_by_project:
            return derivative
        # decay = self.current_time - self.last_time_of_projects[project_id]
        decay = np.exp(-self.beta * (self.current_time - self.last_time_of_projects[project_id]))
        derivative[project_id] += self.user_embedding + decay * self.project_derivative_by_project[project_id]
        return derivative
        # decay = self.num_of_events - self.last_time_of_projects[project_id]
        # return self.user_embedding + decay * self.project_weight[project_id]
