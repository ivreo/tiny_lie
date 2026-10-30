import numpy as np
import matplotlib.pyplot as plt
from lie import Vec3, UnitQuaternion


def main():

    # synthetic data (50 Gaussian corruptions of R_true)
    rng = np.random.default_rng(0)
    R_true = UnitQuaternion.Exp(Vec3(0.5, -0.3, 0.8))
    rots = [R_true.rplus(Vec3(*(0.1 * rng.normal(size=3)))) for _ in range(50)]

    # Estimate Karcher(?) mean
    mu = rots[0]
    step_sizes, errors = [], []
    for _ in range(20):
        # distances
        taus = [rot.rminus(mu) for rot in rots]
        tau = sum(taus, Vec3(0., 0., 0.)) / len(rots)
        # graph data gathering
        step_sizes.append(tau.norm())
        errors.append(mu.rminus(R_true).norm())  
        # update
        mu = mu.rplus(tau)

    print(f"final error to truth: {errors[-1]:.2e}")
    plt.semilogy(errors, marker='o', label='error to true mean')
    plt.semilogy(step_sizes, marker='s', label='step size ‖τ‖')
    plt.xlabel('iteration'); plt.ylabel('geodesic distance (rad)')
    plt.title('Rotation averaging convergence'); plt.legend(); plt.grid(True)
    plt.savefig('averaging_convergence_quat.png', dpi=120)
    print("saved averaging_convergence_quat.png")




if __name__ == "__main__":
    main()
