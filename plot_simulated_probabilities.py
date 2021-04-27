import numpy as np
import matplotlib.pyplot as plt
from simulate_probabilities import x, dims


y = np.loadtxt("simulated_probabilities.txt")

if __name__ == "__main__":
    for dim, yy in zip(dims, y):
        plt.plot(x, yy, label="D={}".format(dim))
    plt.ylim(0, 1)
    plt.xlabel("$r$")
    plt.ylabel("probability of a good separation")
    plt.legend()
    plt.savefig("projected_separations_plot.png", dpi=1000)
