import numpy as np
import hypothesis.strategies as st
from hypothesis import assume
from lie import Vec3, SE3Tangent

@st.composite
def rotation_vectors(draw, max_angle=np.pi - 1e-3) -> Vec3:
    """ Random rotation vector

          tau = theta u 
              with theta in [0, max_angle]
              and   ||u|| = 1

        note: the edge case pi is excluded by default,
              to stay out of the ill-conditioned Log region near pi
              pass max_angle=np.pi to include it
    """
    axis = Vec3(draw(st.floats(-1, 1)), draw(st.floats(-1, 1)), draw(st.floats(-1, 1)))
    assume(axis.norm() > 1e-9)
    theta = draw(st.floats(0, max_angle))
    u = axis.normalized()
    return theta*u
    # TODO this sampling is not uniform on a ball


@st.composite
def se3_tangents(draw, max_angle=np.pi - 1e-3, max_translation=10.0) -> SE3Tangent:
    """ Random se(3) tangent xi = (rho, theta)

        rho: translation part, in [-max_translation, +max_translation]
        theta: rotation part, rotation vector ||theta|| in [0, max_angle]

    """
    m = max_translation
    rho = Vec3(draw(st.floats(-m, m)),
               draw(st.floats(-m, m)),
               draw(st.floats(-m, m)))
    theta = draw(rotation_vectors(max_angle=max_angle))
    return SE3Tangent(rho, theta)
