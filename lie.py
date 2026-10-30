from __future__ import annotations # for signature annotation, '-> Vec3' instead of '-> "Vec3"'
import numpy as np
from math import sqrt
import math 

EPS = 10**(-8)

class Vec3:

    def __init__(self, x, y, z):
        self.v = np.array([x,y,z], dtype=float)

    @property
    def x(self):
        return self.v[0]

    @property
    def y(self):
        return self.v[1]

    @property
    def z(self):
        return self.v[2]

    # called when vec*s
    def __mul__(self, s: float) -> Vec3:
        return Vec3(*(self.v*s))
    # called when s*vec
    __rmul__ = __mul__

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(*(self.v + other.v))

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(*(self.v - other.v))

    def __truediv__(self, s: float) -> Vec3:
        return Vec3(*(self.v/s))

    def __neg__(self) -> Vec3:
        return Vec3(*(-self.v))

    def dot(self, other):
        return float(self.v.dot(other.v))

    def norm(self):
        return float(sqrt(self.dot(self)))

    def normalized(self):
        n = self.norm()
        if n < 1e-12:
            raise ValueError("cannot normalize a zero-length vector")
        return self/n

    def cross(self, other):
        return Vec3(self.y*other.z - self.z*other.y,
                    self.z*other.x - self.x*other.z,
                    self.x*other.y - self.y*other.x)

    # helper functions for testing
    def approx_eq(self, other: Vec3, tol=1e-12) -> bool:
        diff = self.__sub__(other)
        return diff.norm() < tol


class Quaternion:

    def __init__(self, w: float, v: Vec3):
        self._w = w
        self._v = v

    @property
    def w(self):
        return self._w

    @property
    def v(self):
        return self._v

    def norm(self):
        return float(self.w*self.w + self.v.dot(self.v)) ** 0.5

    def conjugate(self):
        return Quaternion(self.w, -self.v)

    def hamilton(self, other: Quaternion) -> Quaternion:
        wout = (  self.w * other.w
                - self.v.dot(other.v))
        vout = (   self.w * other.v
                 + other.w * self.v
                 + self.v.cross(other.v))
        return Quaternion(wout, vout)

    def scaled(self, s:float) -> Quaternion:
        return Quaternion(self.w * s, self.v * s)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return self.hamilton(other)
        elif isinstance(other, (float, int)):
            return self.scaled(other)
        return NotImplemented

    def __rmul__(self, other) -> Quaternion:
        if isinstance(other, (float, int)):
            return self.scaled(other)
        return NotImplemented

    # checking all options are covered:
    #  q*p  q.__mul__(p) Hamilton product
    #  p*q  p.__mul__(q) Hamilton product (not commutative)
    #  q*s  q.__mul__(s) scalar product (the second branch)
    #  s*q  tries s.__mul__(q) not implemented
    #       then calls  q.__rmul__(s) scalar product

    def __add__(self, other: Quaternion) -> Quaternion:
        return Quaternion(self.w+other.w, self.v+other.v)

    def __sub__(self, other:Quaternion) -> Quaternion:
        return Quaternion(self.w-other.w, self.v-other.v)

    def approx_eq(self, other, tol=1e-12) -> bool:
        return (self-other).norm() < tol












class UnitQuaternion(Quaternion):

    def __init__(self, w: float, v: Vec3):

        n = (w*w + v.dot(v)) ** 0.5
        if (n < 1e-12):
            raise ValueError("building a unit quaternion from ~zero norm")
        self._w = w / n
        self._v = v / n  # Using the Vec3 __truediv__


    @classmethod
    def from_quaternion(cls, q: Quaternion) -> UnitQuaternion:
        return cls(q.w, q.v)

    @classmethod
    def identity(cls) -> UnitQuaternion:
        return cls(1.0, Vec3(0.0, 0.0, 0.0))

    @classmethod
    def Exp(cls, tau: Vec3) -> UnitQuaternion:
        theta = tau.norm()
        if theta < EPS: # small angles
            return cls(1-theta*theta/8, tau/2) # Taylor expansions of cos(theta/2) and sin(theta/2)/theta
        else:
            return cls(math.cos(theta/2), tau*math.sin(theta/2)/theta)

    def Log(self) -> Vec3:
        vnorm = self._v.norm()
        if vnorm < EPS: # small angles
            f = 2  # Taylor expansion of theta/sin(theta/2)
        else:
            f = 2*math.atan2(vnorm, self._w)/vnorm
        return f*self._v

    @classmethod # for alternative constructors
    def from_axis_angle(cls, axis: Vec3, theta: float) -> UnitQuaternion:
        return cls.Exp(axis.normalized()*theta)

    def inverse(self):
        return self.conjugate()

    def rotate(self, p: Vec3) -> Vec3:
        pq = self * Quaternion(0,p) * self.conjugate()
        return pq.v

    def to_matrix(self) -> np.ndarray:
        c0 = self.rotate(Vec3(1., 0., 0.)).v
        c1 = self.rotate(Vec3(0., 1., 0.)).v
        c2 = self.rotate(Vec3(0., 0., 1.)).v
        return np.column_stack([c0, c1, c2])

    def rplus(self, tau: Vec3) -> UnitQuaternion:
        q = self * UnitQuaternion.Exp(tau) # Quaternion (not UnitQuaternion)
        return UnitQuaternion.from_quaternion(q)

    def rminus(self, other: UnitQuaternion) -> Vec3:
        q = other.inverse() * self
        return UnitQuaternion.from_quaternion(q).Log()
        # dealing with hamilton returning Quaternion type...
        #   another options is modifying hamilton: with type(self)(w_out, v_out)


class SO3:

    def __init__(self, R: np.ndarray):
        self._R = R
        # something to verify it's an actual rotation matrix?

    def __mul__(self, other):
        return SO3(self._R @ other._R)

    @staticmethod
    def hat(u: Vec3) -> np.ndarray:
        x, y, z = u.v
        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]])

    @staticmethod
    def vee(M: np.ndarray) -> Vec3:
        assert np.allclose(M,-M.T), "vee input is not skew symmetric"
        return Vec3(M[2,1], M[0,2], M[1,0])

    @classmethod
    def Exp(cls, tau: Vec3) -> SO3:
        angle = tau.norm()
        I = np.eye(3)
        if angle < EPS:
            # sin theta / theta -> 1
            # (cos theta - 1) -> 0
            T = cls.hat(tau)
            R = I + T 
        else:
            # rodrigues
            U = cls.hat(tau/angle)
            c = math.cos(angle)
            s = math.sin(angle)
            R = I + s*U + (1-c)*(U@U)
        return cls(R)

    def Log(self) -> Vec3:
        t = self.vee(self._R-self._R.T)/2
        sinA = t.norm()  # assuming sinA >= 0 w.l.g.
        cosA = (self._R.trace()-1)/2
        angle = math.atan2(sinA, cosA)
        if angle < EPS:
            tau = t
        elif EPS < angle < math.pi-EPS:
            tau = angle/sinA*t
        else: # angle ~ pi
            A = (self._R + np.eye(3))/2  # ~= u @ u^T
            max_idx = np.argmax(np.diag(A))
            u = A[:,max_idx]/sqrt(A[max_idx, max_idx])
            tau = angle * Vec3(*u)
        return tau


    def rplus(self, tau: Vec3) -> SO3:
        return self * SO3.Exp(tau)   # R o Exp(tau), R cirlePlus tau

    def rminus(self, other: SO3) -> Vec3:
        return (other.inverse() * self).Log()
        # returns the local-frame tangent
        #    at 'other' that reaches 'self'
        # convention:
        #  - self is the left operand
        #  - Y circleMinus X
        #    Y[self].rminus(X[other])
        #    Log (Xinv o Y)
        #    Log(other.inv * self)

    @classmethod
    def from_axis_angle(cls, axis: Vec3, theta: float) -> SO3:
        return cls.Exp(axis.normalized()*theta)

    def inverse(self) -> SO3:
        return SO3(self._R.T)

    def approx_eq_by_action(self, other: SO3, tol=1e-9) -> bool:
        for e in (Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)):
            if not self.rotate(e).approx_eq(other.rotate(e)):
                return False
        return True

    def approx_eq(self, other: SO3, tol: float = 1e-9) -> bool:
        relative = self.inverse() * other
        return relative.Log().norm() < tol



def compute_frechet_average(rots: list[SO3], iters: int =20) -> SO3:
    mu = rots[0]
    for _ in range(iters):
        taus = [rot.rminus(mu) for rot in rots]
        tau = sum(taus, Vec3(0., 0., 0.)) / len(rots)
        mu = mu.rplus(tau)
    return mu















