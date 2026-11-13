from lie import SE3, SE3Tangent, Vec3
import numpy as np
from hypothesis import given, strategies as st, assume
from conftest import se3_tangents

from scipy.linalg import expm

### SE3tangent conversions roundtrips

@given(xi=se3_tangents())
def test_se3tangent_numpy_roundtrip(xi):
    assert np.allclose(SE3Tangent.from_numpy(xi.to_numpy()).to_numpy(), xi.to_numpy())



### Exp and Log , roundtrip test, closed form compared to matrix exponential

@given(xi=se3_tangents())
def test_se3_exp_log_roundtrip(xi):
    T = SE3.Exp(xi)
    assert np.allclose(SE3.Exp(T.Log()).to_matrix(), T.to_matrix())

@given(xi=se3_tangents())
def test_se3_exp_matches_expm(xi):
    assert np.allclose(SE3.Exp(xi).to_matrix(), expm(xi.hat()))


###  Mult associative, and Inverse

@given(a=se3_tangents(), b=se3_tangents(), c=se3_tangents())
def test_se3_compose_associative(a, b, c):   # (AB)C == A(BC)
    A, B, C = SE3.Exp(a), SE3.Exp(b), SE3.Exp(c)
    X = (A*B)*C
    Y = A*(B*C)
    assert np.allclose(X.to_matrix(), Y.to_matrix())

@given(xi=se3_tangents())
def test_se3_inverse(xi):
    T = SE3.Exp(xi)
    assert np.allclose((T * T.inverse()).to_matrix(), np.eye(4))
    assert np.allclose((T.inverse() * T).to_matrix(), np.eye(4))


### Action
# Test: T.act(p) matches the homogeneous form (T.to_matrix() @ [p; 1])[:3].
@given(t=se3_tangents(),
       px=st.floats(-10, 10), py=st.floats(-10, 10), pz=st.floats(-10, 10))
def test_se3_act_matches_homogeneous(t, px, py, pz):
    T = SE3.Exp(t)
    p = Vec3(px, py, pz)
    p_hom = np.append(p.to_numpy(), 1.0)             # [p; 1], shape (4,)
    expected = (T.to_matrix() @ p_hom)[:3]           # drop the homogeneous 1
    assert np.allclose(T.act(p).to_numpy(), expected)
#TODO consider adding point generation into conftest.py


### right minus, right plus
# round-trip test  X.rplus(xi).rminus(X) == xi (compare via .to_numpy(), or at the group level for the θ≈π safety).
@given(t=se3_tangents(), xi=se3_tangents())
def test_se3_rplus_rminus_roundtrip(t, xi):
    T = SE3.Exp(t)                                   # arbitrary base pose
    recovered = T.rplus(xi).rminus(T)                # (X [+] xi) [-] X  ==  xi
    assert np.allclose(SE3.Exp(recovered).to_matrix(), SE3.Exp(xi).to_matrix())

### Adjoint - checking the definition "conjugation test" ? 
@given(xi=se3_tangents(), t=se3_tangents())
def test_se3_adjoint(t, xi):
    T = SE3.Exp(t) # some base pose
    xip = SE3Tangent.from_numpy(T.adjoint() @ xi.to_numpy())    #  xi' = Ad xi
    T1 = T * SE3.Exp(xi)
    T2 = SE3.Exp(xip) * T
    assert np.allclose(T1.to_matrix(), T2.to_matrix())
