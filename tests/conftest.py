import numpy as np
import hypothesis.strategies as st
from hypothesis import assume
from lie import Vec3

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
