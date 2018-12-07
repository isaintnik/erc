import time
from src.main.python.model import *
from src.main.python.data_generator import *


def generate_synthetic():
    return [
        [USession(0, 10, 15, None, 10), USession(1, 16, 18, None, 5),
         USession(2, 25, 30, None, 10), USession(1, 36, 38, 20, 3), USession(1, 44, 49, 6, 7)],
        [USession(2, 10, 16, None, 8), USession(0, 20, 28, None, 16),
         USession(2, 36, 40, 10, 8), USession(2, 48, 60, 50, 24)]
    ]


def generate_vectors(users_num, projects_num, dim, mean=0.5, std_dev=0.2):
    return np.array([np.random.normal(mean, std_dev, dim) for _ in range(users_num)]), \
           np.array([np.random.normal(mean, std_dev, dim) for _ in range(projects_num)])


def interaction_matrix(users, projects):
    return users @ projects.T


def correct_likelihood_test():
    m = Model(generate_synthetic(), 2, eps=10)
    log_likelihood = m.log_likelihood()
    print(log_likelihood)
    user_derivatives, project_derivatives = m.ll_derivative()
    print(user_derivatives)
    print(project_derivatives)
    m.user_embeddings += np.ones_like(m.user_embeddings) * 0.0001
    second_log_likelihood = m.log_likelihood()
    print(second_log_likelihood)
    print(second_log_likelihood - log_likelihood)


def correct_derivative_test():
    users_num = 2
    projects_num = 2
    dim = 2
    users, projects = generate_vectors(users_num, projects_num, dim)
    users_history = generate_synthetic()
    model = Model(users_history, dim, learning_rate=0.01, eps=10)
    user_embeddings1 = model.user_embeddings
    print("truth")
    print(users)
    print()
    print("init emb")
    print(user_embeddings1)
    # project_embeddings1 = model.get_project_embeddings()
    # print(project_embeddings1)
    users_derivatives, project_derivatives = model.ll_derivative()
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
    other_project_importance = 0.3
    users, projects = generate_vectors(users_num, projects_num, dim, std_dev=0.1)
    X = [StepGenerator(user_embedding=user, project_embeddings=projects, beta=beta,
                       other_project_importance=other_project_importance, max_lifetime=50000, verbose=False)
             .generate_user_steps() for user in users]
    print(len(X[0]))
    print("data generated")
    model = Model2Lambda(X, dim, learning_rate=0.0001, eps=20, beta=beta,
                         other_project_importance=other_project_importance,
                         projects_embeddings_prior=projects)
    # model = Model2UA(X, dim, learning_rate=0.003, eps=30, beta=beta,
    #                  other_project_importance=other_project_importance)
    start_interaction = interaction_matrix(users, projects)
    init_interaction = interaction_matrix(model.user_embeddings, model.project_embeddings)
    for i in range(41):
        if i % 20 == 0:
            inter_norm = np.linalg.norm(
                interaction_matrix(model.user_embeddings, model.project_embeddings) - start_interaction)
            print("{}-th iter, ll = {}, inter_norm = {}".format(i, model.log_likelihood(), inter_norm))
        model.optimization_step()
    end_interaction = interaction_matrix(model.user_embeddings, model.project_embeddings)
    print(np.linalg.norm(start_interaction - init_interaction))
    print(np.linalg.norm(start_interaction - end_interaction))


def init_compare_test():
    users_num = 5
    projects_num = 5
    dim = 2
    beta = 0.001
    other_project_importance = 0.3
    learning_rate = 0.3
    iter_num = 56
    users, projects = generate_vectors(users_num, projects_num, dim, mean=0.5, std_dev=0.2)
    users_init, projects_init = generate_vectors(users_num, projects_num, dim, mean=0.3, std_dev=0.2)
    X = generate_history(users=users, projects=projects, beta=beta, other_project_importance=other_project_importance,
                         max_lifetime=50000)
    print(len(X[0]))
    print("data generated")
    model1 = Model2Lambda(X, dim, learning_rate=learning_rate, eps=20, beta=beta,
                          other_project_importance=other_project_importance,
                          users_embeddings_prior=users_init,
                          projects_embeddings_prior=projects_init)
    model2 = Model2Lambda(X, dim, learning_rate=learning_rate, eps=20, beta=beta,
                          other_project_importance=other_project_importance,
                          users_embeddings_prior=users,
                          projects_embeddings_prior=projects)
    start_interaction = interaction_matrix(users, projects)
    init_interaction = interaction_matrix(model1.user_embeddings, model1.project_embeddings)
    for i in range(iter_num):
        if i % 5 == 0 or i in [1, 2]:
            end_interaction1 = interaction_matrix(model1.user_embeddings, model1.project_embeddings)
            end_interaction2 = interaction_matrix(model2.user_embeddings, model2.project_embeddings)
            inter_norm1 = np.linalg.norm(end_interaction1 - start_interaction)
            inter_norm2 = np.linalg.norm(end_interaction2 - start_interaction)
            print("{}-th iter, ll = {}, inter_norm = {}".format(i, model1.log_likelihood(), inter_norm1))
            print("{}-th iter, ll = {}, inter_norm = {}".format(i, model2.log_likelihood(), inter_norm2))
            print("|m1 - m2| = {}, |m1*c - s| = {}".format(np.linalg.norm(end_interaction2 - end_interaction1),
                  np.linalg.norm(np.mean(start_interaction / end_interaction1) * end_interaction1 - start_interaction)))
            print()
        model1.optimization_step()
        model2.optimization_step()
    end_interaction1 = interaction_matrix(model1.user_embeddings, model1.project_embeddings)
    end_interaction2 = interaction_matrix(model2.user_embeddings, model2.project_embeddings)
    print("|start - init| =", np.linalg.norm(start_interaction - init_interaction))
    print("|start - end_init| =", np.linalg.norm(start_interaction - end_interaction1))
    print("|start - end_true| =", np.linalg.norm(start_interaction - end_interaction2))
    print("coeff =", np.mean(end_interaction1 / start_interaction))
    print(start_interaction)
    print(end_interaction1)
    print(end_interaction2)
    print()
    print(np.linalg.norm(np.mean(start_interaction / end_interaction1) * end_interaction1 - start_interaction))


if __name__ == "__main__":
    np.random.seed(3)
    start_time = time.time()
    # correct_derivative_test()
    # convergence_test()
    # synthetic_test()
    init_compare_test()
    print("time:", time.time() - start_time)
