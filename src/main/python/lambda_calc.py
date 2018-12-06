import copy
import numpy as np
from src.main.python import wheel


class InteractionCalculator:
    def __init__(self, user_embeddings, project_embeddings):
        self.user_embeddings = user_embeddings
        self.project_embeddings = project_embeddings
        self.interactions = user_embeddings @ project_embeddings.T

    # def add(self, user_id, project_id):
    #     if (user_id, project_id) not in self.interactions:
    #         self.interactions[user_id, project_id] = self.user_embeddings[user_id] @ \
    #                                                  self.project_embeddings[project_id].T

    def get_interaction(self, user_id, project_id):
        return self.interactions[user_id, project_id]

    def get_user_supplier(self, user_id):
        return lambda project_id: self.interactions[user_id, project_id]


class UserProjectLambda:
    def __init__(self, user_embedding, dim, beta, interactions_supplier, *, derivative=False, square=False):
        self.numerator = 10
        self.denominator = 10
        self.beta = beta
        self.dim = dim
        # self.square = square
        self.derivative = derivative
        self.user_embedding = user_embedding
        self.user_d_numerator = np.zeros(dim)
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
                    self.project_d_numerators[session_project_id] = np.zeros(self.dim)
                self.project_d_numerators[self.this_project_id] += coeff * self.user_embedding * n_tasks  # * (2 * ua if self.square else 1)

    def get(self):
        cur_lambda = self.numerator / self.denominator / self.avg_time_between_sessions
        if self.derivative:
            user_derivative = self.user_d_numerator / self.denominator / self.avg_time_between_sessions
            # solve problem of few projects
            # maybe we should separate all counting fields of UserProjectLambda (nominator, denominator, derivatives)
            project_derivative = self.project_d_numerators / self.denominator / self.avg_time_between_sessions
            return cur_lambda, user_derivative, project_derivative
        return cur_lambda


class UserLambda:
    def __init__(self, user_embedding, n_projects, beta, other_project_importance, interactions_supplier,
                 derivative=False, square=False):
        self.project_lambdas = {}
        self.beta = beta
        self.derivative = derivative
        self.other_project_importance = other_project_importance
        self.user_embedding = user_embedding
        self.default_lambda = UserProjectLambda(self.user_embedding, n_projects, self.beta, interactions_supplier,
                                                derivative=derivative, square=square)

    def update(self, project_embedding, session, delta_time):
        if session.pid not in self.project_lambdas:
            # make other copy, not deep for non copy interactions_supplier
            self.project_lambdas[session.pid] = copy.deepcopy(self.default_lambda)
            self.project_lambdas[session.pid].set_project_id(session.pid)
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


def calc_lambdas(user_id, project_id, history, user_embedding, dim, beta, interactions, projects_embeddings):
    upl = UserProjectLambda(user_embedding, dim, beta, interactions.get_user_supplier(user_id))
    ans = []
    for session in history:
        upl.update(projects_embeddings[session.pid], session.pid, session.n_tasks, None,
                   1 if project_id == session.pid else 0.3)
        ans.append(upl.get())
    return np.array(ans)


def calc_lambdas_native(user_id, project_id, project_ids, n_tasks, time_deltas, user_embedding, dim, beta,
                        interactions, projects_embeddings):
    out_lambdas = np.zeros(len(project_ids), dtype=np.float64)
    wheel.calc_lambdas(project_id, user_embedding, projects_embeddings, dim, beta, interactions, False, 0.3,
                       project_ids, n_tasks, time_deltas, out_lambdas, None, None)
    return out_lambdas
