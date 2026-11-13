from lie import SE3, SE3Tangent
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

