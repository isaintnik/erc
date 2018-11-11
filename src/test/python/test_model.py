from src.main.python import model


TESTING_USER_HISTORIES = [
    [(10000, 15000, 0, 10), (16000, 18000, 1, 5), (25000, 30000, 0, 10)],
    [(12000, 16000, 0, 8), (20000, 28000, 1, 16), (36000, 40000, 1, 8), (48000, 60000, 1, 24)]
]
STEP = 1e-4


def test_working():
    m = model.Model(TESTING_USER_HISTORIES, 5)
    m.log_likelihood()
    m.calc_derivative()


def test_correct():
    m = model.Model(TESTING_USER_HISTORIES, 5, beta=8e-3)
    log_likelihood = m.log_likelihood()
    user_derivatives, project_derivatives = m.calc_derivative()
    print(log_likelihood)
    print(user_derivatives)
    print(project_derivatives)
    m.user_embeddings[1][0] += STEP
    second_log_likelihood = m.log_likelihood()
    print(second_log_likelihood - log_likelihood)


if __name__ == '__main__':
    test_working()
