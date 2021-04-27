import numpy as np


def separation_trials(a, num_trials=100_000):
    dim = len(a)
    pts = a * (np.random.randn(num_trials, dim)) ** 2
    total = np.expand_dims(pts.sum(axis=1), axis=1)
    successes = np.sum(pts > total / 2, axis=0)
    return successes, num_trials


def total_separation_prob(a, num_trials=100_000):
    separations, trials = separation_trials(a, num_trials)
    return separations.sum() / trials


x = np.linspace(1, 2)
dims = np.array([3, 5, 10, 15, 100])

if __name__ == "__main__":
    y = np.zeros((len(dims), len(x)))

    np.random.seed(1234)
    for i in range(len(dims)):
        print(f"simulating dimension {dims[i]}...", end="")
        a = np.expand_dims(x, 1) ** np.arange(dims[i])
        y[i] = np.apply_along_axis(total_separation_prob, 1, a, num_trials=1_000_000)
        print("done.")

    np.savetxt("simulated_probabilities.txt", y)
