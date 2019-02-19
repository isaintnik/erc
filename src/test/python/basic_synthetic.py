import time
from src.main.python.model import Model
from src.main.python.data_generator import *


def generate_vectors(users_num, projects_num, dim, mean=0.5, std_dev=0.2):
    return {user_id: np.random.normal(mean, std_dev, dim) for user_id in range(users_num)}, \
           {project_id: np.random.normal(mean, std_dev, dim) for project_id in range(projects_num)}


def interaction_matrix(users, projects):
    users_vecs = np.array([users[key] for key in sorted(list(users.keys()), key=lambda x: str(x))])
    projects_vecs = np.array([projects[key] for key in sorted(list(projects.keys()), key=lambda x: str(x))])
    return users_vecs @ projects_vecs.T


def correct_likelihood_test():
    data = generate_synthetic()
    m = ModelExpLambda(data, 2, eps=10)
    log_likelihood = m.log_likelihood(data)
    print(log_likelihood)
    user_derivatives, project_derivatives = m.ll_derivative(data)
    print(user_derivatives)
    print(project_derivatives)
    m.user_embeddings += np.ones_like(m.user_embeddings) * 0.0001
    second_log_likelihood = m.log_likelihood(data)
    print(second_log_likelihood)
    print(second_log_likelihood - log_likelihood)


def correct_derivative_test():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic()
    model = ModelExpLambda(dim, eps=10)
    user_embeddings1 = model.user_embeddings
    print("truth")
    print(users)
    print()
    print("init emb")
    print(user_embeddings1)
    # project_embeddings1 = model.get_project_embeddings()
    # print(project_embeddings1)
    users_derivatives, project_derivatives = model.ll_derivative(users_history)
    print()
    print("derivative")
    print(users_derivatives)
    # print(project_derivatives)
    users_delta = 0.001 * (users - user_embeddings1)
    print()
    print("delta")
    print(users_delta)
    model.user_embeddings += users_delta
    users_derivatives, project_derivatives = model.ll_derivative()
    # projects_delta = 0.001 * (projects - project_embeddings1)
    # print(projects_delta)
    # for (user_derivative, user_delta) in zip(users_derivatives, users_delta):
    #     assert np.abs(user_derivative - user_delta) < 0.0001
    # for (project_derivative, project_delta) in zip(project_derivatives, projects_delta):
    #     assert np.abs(project_derivative - project_delta) < 0.0001


def test_convergence():
    dim = 3
    users_history = generate_synthetic()
    # model = Model2UA(users_history, dim, learning_rate=0.003, eps=10)
    model = Model2Lambda(users_history, dim, beta=0.001, learning_rate=0.005, eps=10)
    for i in range(101):
        if i % 20 == 0:
            print("{}-th iter, ll = {}".format(i, model.log_likelihood()))
        model.optimization_step()


def synthetic_test():
    users_num = 2
    projects_num = 2
    dim = 2
    beta = 0.001
    eps = 1
    other_project_importance = 0.3
    learning_rate = 3
    lambda_transform = lambda x: np.exp(x)
    lambda_derivative = lambda x: np.exp(x)
    users_embedding, project_embeddings = generate_vectors(users_num, projects_num, dim, mean=0.1, std_dev=0.01)
    data = generate_sythetic(users_embedding, project_embeddings, beta, other_project_importance,
                             lambda_transform=lambda_transform, max_lifetime=50)
    print("data generated, |events| =", len(data))
    model = Model(dim, beta, eps, other_project_importance, lambda_transform=lambda_transform,
                  lambda_derivative=lambda_derivative)
    # model = Model2UA(X, dim, learning_rate=0.003, eps=30, beta=beta,
    #                  other_project_importance=other_project_importance)
    print("start_ll =", model.log_likelihood(data))
    start_interaction = interaction_matrix(users_embedding, project_embeddings)
    init_interaction = interaction_matrix(model.user_embeddings, model.project_embeddings)
    for i in range(7):
        model.fit(data, learning_rate, iter_num=1)
        inter_norm = np.linalg.norm(
                        interaction_matrix(model.user_embeddings, model.project_embeddings) - start_interaction)
        print("{}-th iter, ll = {}, inter_norm = {}".format(i, model.log_likelihood(data), inter_norm))
        # print(interaction_matrix(model.user_embeddings, model.project_embeddings))
    # for i in range(41):
    #     if i % 20 == 0:
    #         inter_norm = np.linalg.norm(
    #             interaction_matrix(model.user_embeddings, model.project_embeddings) - start_interaction)
    #         print("{}-th iter, ll = {}, inter_norm = {}".format(i, model.log_likelihood(), inter_norm))
    #     model.optimization_step()
    end_interaction = interaction_matrix(model.user_embeddings, model.project_embeddings)
    print(np.linalg.norm(start_interaction - init_interaction))
    print(np.linalg.norm(start_interaction - end_interaction))
    print()
    print(start_interaction)
    print()
    print(end_interaction)


def relu(x):
    return x if x > 0 else 0


def relu_d(x):
    return 1 if x > 0 else 0


def init_compare_test():
    users_num = 2
    projects_num = 2
    dim = 2
    beta = 0.001
    eps = 1
    other_project_importance = 0.1
    embedding_mean = 0.2
    std_dev = 0.3
    default_lambda = embedding_mean ** 2
    # lambda_transform = lambda x: x ** 2
    # lambda_derivative = lambda x: 2 * x
    # lambda_transform = relu
    # lambda_derivative = relu_d
    lambda_transform = abs
    lambda_derivative = np.sign
    learning_rate = 0.02
    inner_iter_num = 3
    outer_iter_num = 20
    user_embedding, project_embeddings = generate_vectors(users_num, projects_num, dim, mean=embedding_mean,
                                                          std_dev=std_dev)
    users_init, projects_init = generate_vectors(users_num, projects_num, dim, mean=embedding_mean, std_dev=std_dev)
    data = generate_sythetic(user_embedding, project_embeddings, beta, other_project_importance,
                             default_lambda=default_lambda, lambda_transform=lambda_transform, max_lifetime=10000)
    print("data generated, |events| =", len(data))
    model_random = Model(dim, beta, eps, other_project_importance, lambda_transform=lambda_transform,
                   lambda_derivative=lambda_derivative, users_embeddings_prior=users_init,
                   projects_embeddings_prior=projects_init)
    model_true = Model(dim, beta, eps, other_project_importance, lambda_transform=lambda_transform,
                   lambda_derivative=lambda_derivative, users_embeddings_prior=user_embedding,
                   projects_embeddings_prior=project_embeddings)
    # print("start_ll_random = {}, start_ll_true = {}".format(model_random.log_likelihood(data),
    #                                                         model_true.log_likelihood(data)))
    start_interaction = interaction_matrix(user_embedding, project_embeddings)
    init_interaction = interaction_matrix(model_random.user_embeddings, model_random.project_embeddings)
    for i in range(outer_iter_num):
        end_interaction1 = interaction_matrix(model_random.user_embeddings, model_random.project_embeddings)
        end_interaction2 = interaction_matrix(model_true.user_embeddings, model_true.project_embeddings)
        inter_norm1 = np.linalg.norm(end_interaction1 - start_interaction)
        inter_norm2 = np.linalg.norm(end_interaction2 - start_interaction)
        print("random: {}-th iter, ll = {}, inter_norm = {}".format(i, model_random.log_likelihood(data), inter_norm1))
        print("  true: {}-th iter, ll = {}, inter_norm = {}".format(i, model_true.log_likelihood(data), inter_norm2))
        print("|m1 - m2| = {}, |m1*c - s| = {}".format(np.linalg.norm(end_interaction2 - end_interaction1),
              np.linalg.norm(np.mean(start_interaction / end_interaction1) * end_interaction1 - start_interaction)))
        print()
        model_random.fit(data, learning_rate, iter_num=inner_iter_num, verbose=False)
        model_true.fit(data, learning_rate, iter_num=inner_iter_num, verbose=False)
        # for user_id in user_embedding:
        #     print("user_id = {}, lambdas = {}".format(user_id,
        #           ", ".join(["({}, {})".format(pid, model_random.get_applicable(data).get_lambda(user_id, pid))
        #                      for pid in project_embeddings])))
    end_interaction_random = interaction_matrix(model_random.user_embeddings, model_random.project_embeddings)
    end_interaction_true = interaction_matrix(model_true.user_embeddings, model_true.project_embeddings)
    print("|start - init| =", np.linalg.norm(start_interaction - init_interaction))
    print("|start - end_random| =", np.linalg.norm(start_interaction - end_interaction_random))
    print("|start - end_true| =", np.linalg.norm(start_interaction - end_interaction_true))
    print("coeff_random =", np.mean(end_interaction_random / start_interaction))
    print("coeff_true =", np.mean(end_interaction_true / start_interaction))
    print()
    print(start_interaction)
    print()
    print(end_interaction_random)
    print()
    print(end_interaction_true)
    print()
    print(np.linalg.norm(np.mean(start_interaction / end_interaction_random) * end_interaction_random - start_interaction))


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    # correct_derivative_test()
    # convergence_test()
    # synthetic_test()
    init_compare_test()
    print("time:", time.time() - start_time)
