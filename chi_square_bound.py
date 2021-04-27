import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp2f1, gamma


def prob(r, D):
    """
    This formula was found using Mathematica.
    """
    return (
        r ** ((D - 1) / 2)
        * gamma(D / 2)
        * hyp2f1((D - 1) / 2, D / 2, (D + 1) / 2, -r)
        / np.sqrt(np.pi)
        / gamma((D + 1) / 2)
    )


r = np.linspace(1, 2)
D = np.array([3, 5, 10, 15, 100])
rr, dd = np.meshgrid(r, D)
pp = prob(rr, dd)

for p, d in zip(pp, D):
    plt.plot(r, p, label=f"D={d}")
plt.legend()
plt.savefig("chi_square_bound.png")
