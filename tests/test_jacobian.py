from lie import Vec3, SO3
import numpy as np
from hypothesis import given, strategies as st, assume
from tests.conftest import rotation_vectors





def _numerical_left_jacobian(tau: Vec3, h: float = 1e-6) -> np.ndarray: 
    """ Left Jacobian computed via finite differences in the tangent space

        J_l is the Jacobian around 0 of the R^3 -> R^3 function

           g(e) = Exp(tau+e) [-]_L Exp(tau)   (i.e. Exp(tau+e) = g(e) [+]_L Exp(tau))

        i.e. the 3x3 matrix in first order approximation

           g(e) ~= J_l @ e + O(||e||^2)
    """
    R = SO3.Exp(tau)
    basis = (Vec3(h, 0, 0), Vec3(0, h, 0), Vec3(0, 0, h)) # increments along the natural basis
    # J_l = \partial g/\partial_e |_0  ( definition)
    #   column_i \approx g(h e_i)/h    (finite difference along each axis)
    cols = [((SO3.Exp(tau + e) * R.inverse()).Log() / h).to_numpy() for e in basis]
    return np.column_stack(cols)


def _numerical_right_jacobian(tau: Vec3, h: float = 1e-6) -> np.ndarray: 
    """ Right Jacobian computed via finite differences in the tangent space

        J_r is the Jacobian around 0 of the R^3 -> R^3 function

           g(e) = Exp(tau+e) [-]_R Exp(tau)   (i.e. Exp(tau+e) = Exp(tau) [+]_R g(e)

        i.e. the 3x3 matrix in first order approximation

           g(e) ~= J_r @ e + O(||e||^2)
    """
    R = SO3.Exp(tau)
    basis = (Vec3(h, 0, 0), Vec3(0, h, 0), Vec3(0, 0, h)) # increments along the natural basis
    cols = [ (SO3.Exp(tau+e).rminus(R)/h).to_numpy() for e in basis]
    ## which means
    # cols = [((R.inverse() * SO3.Exp(tau + e)).Log() / h).v for e in basis]   
    return np.column_stack(cols)

@given(tau=rotation_vectors())
def test_left_jacobian_inverse(tau):
    assert np.allclose(SO3.left_jacobian(tau) @ SO3.left_jacobian_inv(tau), np.eye(3))

@given(tau=rotation_vectors())
def test_right_jacobian_inverse(tau):
    assert np.allclose(SO3.right_jacobian(tau) @ SO3.right_jacobian_inv(tau), np.eye(3))

@given(tau=rotation_vectors())
def test_left_jacobian_matches_numerical(tau):
    assert np.allclose(SO3.left_jacobian(tau), _numerical_left_jacobian(tau), atol=1e-4)

@given(tau=rotation_vectors())
def test_right_jacobian_matches_numerical(tau):
    assert np.allclose(SO3.right_jacobian(tau), _numerical_right_jacobian(tau), atol=1e-4)

@given(tau=rotation_vectors())
def test_right_is_left_negated(tau):                      # J_r(tau) = J_l(−tau)
    assert np.allclose(_numerical_right_jacobian(tau), _numerical_left_jacobian(-tau), atol=1e-4)

@given(tau=rotation_vectors())
def test_left_is_adjoint_of_right(tau):                   # J_l = Ad_R J_r = R · J_r
    R = SO3.Exp(tau).to_matrix()
    assert np.allclose(SO3.left_jacobian(tau), R @ SO3.right_jacobian(tau))

