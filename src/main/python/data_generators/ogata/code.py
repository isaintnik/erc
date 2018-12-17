import random
import math
import scipy.special


class IntensityFunction:
    def __init__(self, alphs, betta, mu):
        self.alphs = alphs
        self.betta = betta
        self.mu = mu


class IntensityFunctionWithTimes:
    def __init__(self, intensity):
        self.times = [0]
        self.intensity = intensity

    def add_time(self, t):
        self.times.append(t)

    def get_times(self):
        return self.times[1:]

    def __call__(self, considered_time=None):
        if considered_time:
            self.times.append(considered_time)
        R = [[0] * len(self.times) for j in range(len(self.intensity.alphs))]
        # S need for log-likelihood function
        # S = [lambda t: (1 - math.exp(-self.betta * t)) / self.betta]
        A = []

        for j in range(len(self.intensity.alphs)):
            for i in range(1, len(self.times)):
                A.append(lambda t:
                         t ** k * math.exp(-self.intensity.betta * t))
                sum_ = 0
                for k in range(0, j + 1):
                    sum_ += (A[j - k](self.times[i] - self.times[i - 1]) *
                             R[k][i-1] *
                             scipy.special.binom(j, k))
                R[j][i] = A[j](self.times[i] - self.times[i - 1]) + sum_
            # S.append(lambda t: ((j + 1) * S[-1](t) - A[j+1](t)) / self.betta)

        result = self.intensity.mu
        for index in range(len(self.intensity.alphs)):
            result += self.intensity.alphs[index] * R[j][-1]
        if considered_time:
            self.times.pop()
        return result


def generate_one_user(T, intensity):
    times = generate_times_hawkes_self_exciting_process(T, intensity)
    return times  # without projects


def generate_times_hawkes_self_exciting_process(T, intensity_with_times):
    # now, max_jump_size is a magic constant
    # the jump size at each point is not larger than max_jump_size
    max_jump_size = 1
    # piecewise constant intensity function
    pc_intensity = intensity_with_times.intensity.mu
    s = 0
    U = random.random()
    u = -math.log(U / pc_intensity)
    if u > T:
        return []
    intensity_with_times.add_time(u)
    while True:
        pc_intensity = intensity_with_times() + max_jump_size
        while True:
            U = random.random()
            u = -math.log(U / pc_intensity)
            s = s + u
            if s > T:
                return intensity_with_times.get_times()
            U = random.random()
            if U <= intensity_with_times(s) / pc_intensity:
                intensity_with_times.add_time(s)
                break
            pc_intensity = intensity_with_times(s)


def generate_data(T, intensity, num_of_users):
    return {user_id:
            generate_one_user(T,
                              IntensityFunctionWithTimes(intensity))
            for user_id in range(0, num_of_users)}


if __name__ == "__main__":
    random.seed(1)
    T = 100
    intensity_function_from_article = IntensityFunction([0.045, -0.300, 0.50],
                                                        1.1,
                                                        0.7)
    num_of_users = 1
    print(generate_data(T, intensity_function_from_article, 1))
