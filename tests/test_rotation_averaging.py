from lie import Vec3, Quaternion, UnitQuaternion, compute_frechet_average
import numpy as np

def test_frechet_average():
    rng = np.random.default_rng(0)
    #R_true = SO3.Exp(Vec3(0.5, -0.3, 0.8))
    R_true = UnitQuaternion.Exp(Vec3(0.5, -0.3, 0.8))
    noisy = [R_true.rplus(Vec3(*(0.05 * rng.normal(size=3)))) for _ in range(100)]
    R_est = compute_frechet_average(noisy)
    error = R_est.rminus(R_true).norm()
    assert error < 0.02,  f"recovered mean off by {error}"
