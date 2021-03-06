from src.main.python.user_lambda import UserLambda


class LambdaStrategy:
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance):
        self.interaction_calculator = interaction_calculator
        self.prev_user_action_time = {}
        self.user_lambdas = {user_id: UserLambda(user_embeddings[user_id], project_embeddings, beta,
                                                 other_project_importance,
                                                 self.interaction_calculator.get_user_supplier(user_id))
                             for user_id in user_embeddings.keys()}

    def get_lambda(self, user_id, project_id):
        raise NotImplementedError()

    def get_lambda_user_derivative(self, user_id, project_id):
        raise NotImplementedError()

    def get_lambda_project_derivative(self, user_id, project_id):
        raise NotImplementedError()

    def accept(self, event):
        raise NotImplementedError()


class LookAheadLambdaStrategy(LambdaStrategy):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance)

    def get_lambda(self, user_id, project_id):
        return self.user_lambdas[user_id].get_lambda(project_id)

    def get_lambda_user_derivative(self, user_id, project_id):
        return self.user_lambdas[user_id].get_lambda_user_derivative(project_id)

    def get_lambda_project_derivative(self, user_id, project_id):
        return self.user_lambdas[user_id].get_lambda_project_derivative(project_id)

    def accept(self, event):
        time_delta = 0
        if event.start_ts is not None and event.uid in self.prev_user_action_time:
            time_delta = event.start_ts - self.prev_user_action_time[event.uid]
        self.user_lambdas[event.uid].update(event.pid, time_delta)
        self.prev_user_action_time[event.uid] = event.start_ts


class NotLookAheadLambdaStrategy(LambdaStrategy):
    def __init__(self, user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance):
        super().__init__(user_embeddings, project_embeddings, interaction_calculator, beta, other_project_importance)
        self.saved_lambdas = {user_id: {} for user_id in user_embeddings.keys()}
        self.saved_lambdas_user_derivative = {user_id: {} for user_id in user_embeddings.keys()}
        self.saved_lambdas_project_derivative = {user_id: {} for user_id in user_embeddings.keys()}

    def get_lambda(self, user_id, project_id):
        if project_id not in self.saved_lambdas[user_id]:
            self.saved_lambdas[user_id][project_id] = self.user_lambdas[user_id].get_lambda(project_id)
        return self.saved_lambdas[user_id][project_id]

    def get_lambda_user_derivative(self, user_id, project_id):
        if project_id not in self.saved_lambdas_user_derivative[user_id]:
            self.saved_lambdas_user_derivative[user_id][project_id] = \
                self.user_lambdas[user_id].get_lambda_user_derivative(project_id)
        return self.saved_lambdas_user_derivative[user_id][project_id]

    def get_lambda_project_derivative(self, user_id, project_id):
        if project_id not in self.saved_lambdas_project_derivative[user_id]:
            self.saved_lambdas_project_derivative[user_id][project_id] = \
                self.user_lambdas[user_id].get_lambda_project_derivative(project_id)
        return self.saved_lambdas_project_derivative[user_id][project_id]

    def accept(self, event):
        user_id = event.uid
        project_id = event.pid
        time_delta = 0
        if event.start_ts is not None and event.uid in self.prev_user_action_time:
            time_delta = event.start_ts - self.prev_user_action_time[event.uid]
        self.user_lambdas[user_id].update(project_id, time_delta)
        self.saved_lambdas[user_id][project_id] = self.user_lambdas[user_id].get_lambda(project_id)
        self.saved_lambdas_user_derivative[user_id][project_id] = \
            self.user_lambdas[user_id].get_lambda_user_derivative(project_id)
        self.saved_lambdas_project_derivative[user_id][project_id] = \
            self.user_lambdas[user_id].get_lambda_project_derivative(project_id)
        self.prev_user_action_time[user_id] = event.start_ts
