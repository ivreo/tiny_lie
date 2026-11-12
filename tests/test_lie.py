from lie import Vec3, Quaternion, UnitQuaternion, SO3, compute_frechet_average
import numpy as np
from math import cos, sin, pi
from hypothesis import given, strategies as st, assume
import pytest
from tests.conftest import rotation_vectors


def test_hamilton_associativity():
    p1 = Quaternion(0.4, Vec3(0.3, 0.56, 2.8))
    p2 = Quaternion(0.6, Vec3(0.2, 1.26, 1.8))
    p3 = Quaternion(0.7, Vec3(0.1, 2.06, 0.8))
    q1 = (p1*p2)*p3
    q2 = p1*(p2*p3)
    assert q1.approx_eq(q2)

def test_conjugate_is_antiautomorphism():
    p1 = Quaternion(0.4, Vec3(0.3, 0.56, 2.8))
    p2 = Quaternion(0.6, Vec3(0.2, 1.26, 1.8))
    q1 = (p1*p2).conjugate()
    q2 = (p2.conjugate())*(p1.conjugate())
    assert q1.approx_eq(q2)

def test_matrix_agrees_with_sandwich():
    q = UnitQuaternion.from_axis_angle(Vec3(1,1,1), 0.7)
    p = Vec3(2.0, -1.0, 0.5)
    via_matrix   = q.to_matrix() @ p.v
    via_sandwich = q.rotate(p).v
    assert np.allclose(via_matrix, via_sandwich)

def test_matrix_is_a_rotation():
    q = UnitQuaternion(0.7, Vec3(2,3,4.))
    R = q.to_matrix()
    assert np.allclose(R.T @ R, np.eye(3))

def test_half_coverage():
    theta = .3*pi
    q = UnitQuaternion.from_axis_angle(Vec3(0,0,1), theta)
    R1 = q.to_matrix()
    c = cos(theta)
    s = sin(theta)
    R2 = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
    assert np.allclose(R1,R2)


def test_rodrigues():
    theta = .3*pi
    c = cos(theta)
    s = sin(theta)
    R1 = SO3.from_axis_angle(Vec3(0,0,1), theta)._R
    R2 = np.array([[c, -s, 0],[s, c, 0],[0,0,1]])
    assert np.allclose(R1,R2)


def random_axis_angle(rng):
    axis = Vec3(*rng.normal(size=3))
    while axis.norm() < 1e-6:                 # guard the ~zero axis
        axis = Vec3(*rng.normal(size=3))
    theta = rng.uniform(-np.pi, np.pi)
    return axis, theta

def test_quaternion_matches_rodrigues():
    rng = np.random.default_rng(42)
    for _ in range(200):
        axis, theta = random_axis_angle(rng)
        R_quat = UnitQuaternion.from_axis_angle(axis, theta).to_matrix()
        R_rod  = SO3.from_axis_angle(axis, theta)._R
        assert np.allclose(R_quat, R_rod), f"mismatch at axis={axis.v}, theta={theta}"

# testing hypothesis
@given(
    ax=st.floats(-1,1), ay=st.floats(-1,1), az=st.floats(-1,1),
    theta=st.floats(-np.pi, np.pi),
)
def test_quat_matches_rodrigues_hyp(ax, ay, az, theta):
    axis = Vec3(ax, ay, az)
    if axis.norm() < 1e-6:
        return                                 # skip degenerate
    assert np.allclose(
        UnitQuaternion.from_axis_angle(axis, theta).to_matrix(),
        SO3.from_axis_angle(axis, theta)._R,
    )



@given(tau=rotation_vectors())
def test_Log_of_Exp(tau):
    t = SO3.Exp(tau).Log()
    R = SO3.Exp(tau)
    assert R.approx_eq(SO3.Exp(t))          # round-trip through the group

@given(tau=rotation_vectors())
def test_Exp_of_Log(tau):
    R = SO3.Exp(tau)
    assert R.approx_eq(SO3.Exp(R.Log()))


@pytest.mark.parametrize("theta", [
    0.0,            # exact identity
    1e-12,          # sub-epsilon - small-angle branch
    1e-7,           # near the small-angle threshold
    np.pi - 1e-7,   # just below pi
    np.pi,          # exact pi - the axis-from-symmetric-part branch
])
@pytest.mark.parametrize("axis", [
    Vec3(1.0, 0.0, 0.0),
    Vec3(0.0, 1.0, 0.0),
    Vec3(0.0, 0.0, 1.0),
    Vec3(1.0, 1.0, 1.0), # off-axis - catches bugs coordinate axes hide
])
def test_Log_Exp_on_corners(axis, theta):
    tau = axis.normalized()*theta
    R = SO3.Exp(tau)
    t = R.Log()
    if abs(theta - np.pi) < 1e-9:
        assert R.approx_eq(SO3.Exp(t))          # round-trip through the group
    else:
        assert t.approx_eq(tau)


# (X \oplus \tau) \ominus X == \tau
@given(
    # base rotation
    bx=st.floats(-3, 3), by=st.floats(-3, 3), bz=st.floats(-3, 3),
    # perturbation direction + bounded angle
    px=st.floats(-1, 1), py=st.floats(-1, 1), pz=st.floats(-1, 1),
    angle=st.floats(0, np.pi - 1e-6),
)
def test_rplus_rminus_roundtrip(bx, by, bz, px, py, pz, angle):
    X = SO3.Exp(Vec3(bx, by, bz))    # arbitrary base rotation
    pdir = Vec3(px, py, pz)
    assume(pdir.norm() > 1e-9)
    tau = pdir.normalized() * angle
    # (X \oplus \tau) \ominus X == \tau
    recovered = X.rplus(tau).rminus(X)
    assert recovered.approx_eq(tau, tol=1e-7), f"rplus then rminus failed: {recovered.v} vs {tau.v}"

