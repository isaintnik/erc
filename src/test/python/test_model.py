from src.main.python import model


def test_working():
    users_history = [
        [(10, 15, 1, 10), (16, 18, 2, 5), (25, 30, 1, 10)],
        [(12, 16, 1, 8), (20, 28, 2, 16)]
    ]
    m = model.Model(users_history, 5)
    m.log_likelihood()
    m.calc_derivative()


if __name__ == '__main__':
    test_working()
