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

    def to_numpy(self) -> np.ndarray:
        return self.v.copy()  # (3,)

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

    def to_matrix(self) -> np.ndarray:
        return self._R.copy()

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
        """ Exponential map  R^3 -> SO(3):  rotation vector -> rotation.

            Takes tau = theta * u (unit axis u, angle theta = ||tau||) and returns
            the rotation by theta around u:

                R = Exp(tau) = exp(hat(tau))

            with hat(tau) the skew-symmetric matrix, element of so(3), the Lie algebra of SO(3)

            Closed form via the Rodrigues formula:

                R  = I + sin(theta) hat(u) + (1 - cos(theta)) hat(u)^2

            Inverse of Log.  Exp(0) = identity. Exp is many-to-one
        """
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
        """ Logarithmic map SO(3) ->  R^3: rotation -> rotation vector

            Returns a rotation vector tau = theta * u, such that Exp(tau) == self.

            Principal value: theta in [0, pi], the smallest rotation that
            reaches self.

            note: At theta = pi the axis sign is ambiguous.

            algo: uses the particular structure of R (as visible in the Rodrigues formula)
            to recover the angle and the axis.

            Inverse of Exp (up to 2pi periodicity).
        """
        t = self.vee(self._R-self._R.T)/2
        sinA = t.norm()  # forces angle in [0,pi].
        cosA = (self._R.trace()-1)/2
        angle = math.atan2(sinA, cosA) # more numerically stable
        if angle < EPS:
            # angle/sinA -> 1 as angle -> 0
            tau = t
        elif EPS <= angle < math.pi-EPS:
            tau = angle/sinA*t
        else: # angle ~ pi
            # Rodrigues reduces to R = I + 0 + 2* hat(u)*hat(u)
            # Using hat(u)*hat(u) = u * u^T - I
            # u* u^T = (R+I)/2
            A = (self._R + np.eye(3))/2 
            # Any column would do for the axis
            # we pick the one with largest norm for stability
            max_idx = np.argmax(np.diag(A))
            u = A[:,max_idx]/np.sqrt(A[max_idx, max_idx])
            tau = angle * Vec3(*u)
        return tau

    def rplus(self, tau: Vec3) -> SO3:
        ''' Right box-plus: the result of applying an increment to the rotation 'self'

            returns the rotation reached by applying a small
            rotation increment on the RIGHT of self (i.e. in its own body/local frame).

              Y = R [+]_R tau = R o Exp(tau)

            Inverse of rminus: self.rplus(tau).rminus(self) == tau.
        '''
        return self * SO3.Exp(tau)

    def rminus(self, other: SO3) -> Vec3:
        """ Right box-minus: the body-frame increment from 'other' to 'self'.

             returns the tangent vector tau such that

                self = other [+]_R tau = other o Exp(tau)   
                # tau = Log(inv(other) o self)

             i.e. the small rotation that, applied on the RIGHT of `other`
             (i.e. in its own body/local frame), turns `other` into `self`.

             The result lives in the tangent space based at `other`;
             tau -> 0 as self -> other.

             Inverse of rplus:  other.rplus(self.rminus(other)) == self.
        """
        return (other.inverse() * self).Log()

    @classmethod
    def left_jacobian(cls, tau: Vec3) -> np.ndarray:
        """ Left Jacobian J_l(tau) of SO(3) 

            J_l is the Jacobian around 0 of the R^3 -> R^3 function

                g(e) = Exp(tau+e) [-]_L Exp(tau)   (i.e. Exp(tau+e) = g(e) [+]_L Exp(tau))
            
            i.e. the 3x3 matrix in first order approximation

                g(e) ~= J_l @ e + O(||e||^2)

            Closed form:

                J_l = I + (1-cos t)/t^2 [tau]_x + (t-sin t)/t^3 [tau]_x^2 ,   t = ||tau||

        """
        t = tau.norm()  # angle
        Phi = cls.hat(tau)
        I = np.eye(3)
        if t < EPS:                        # 0/0 in the coefficients as t -> 0
            a = 0.5 - t*t/24.0             # (1-cos t)/t^2  -> 1/2 - t^2/24
            b = 1.0/6.0 - t*t/120.0        # (t-sin t)/t^3  -> 1/6 - t^2/120
        else:
            a = (1.0 - math.cos(t)) / (t*t)
            b = (t - math.sin(t)) / (t**3)
        return I + a*Phi + b*(Phi @ Phi)

    @classmethod
    def left_jacobian_inv(cls, tau: Vec3) -> np.ndarray:
        """Inverse left Jacobian J_l^{-1}(theta) of SO(3)

            Closed form:

            J_l^{-1} = I - 1/2 [tau]_x
                         + ( 1/t^2 - (1+cos t)/(2 t sin t) ) [tau]_x^2 ,   t = ||tau||

            note on how to derive it:
                Cayley-Hamilton gives [tau]_x^3 = -t^2 [tau]_x, so the algebra
                generated by [tau]_x closes: {I, [tau]_x, [tau]_x^2} is a basis.
                Hence J_l and J_l^{-1} are both of the form a I + b [tau]_x + c [tau]_x^2.
                Assume that shape for J_l^{-1}, impose J_l J_l^{-1} = I, match coefficients.
        """
        t = tau.norm()
        Phi = cls.hat(tau)
        I = np.eye(3)
        if t < EPS:                        # 1/t^2 - (1+cos t)/(2 t sin t) -> 1/12 + t^2/720
            c = 1.0/12.0 + t*t/720.0
        else:
            c = 1.0/(t*t) - (1.0 + math.cos(t)) / (2.0*t*math.sin(t))
        return I - 0.5*Phi + c*(Phi @ Phi)

    @classmethod
    def right_jacobian(cls, tau: Vec3) -> np.ndarray:
        """Right Jacobian J_r(theta) of SO(3)
            J_r = J_l^T
        """
        return cls.left_jacobian(tau).T

    @classmethod
    def right_jacobian_inv(cls, tau: Vec3) -> np.ndarray:
        """Inverse right Jacobian J_r^{-1}(theta) of SO(3)
            J_r^{-1} = J_l^{-1}^T
        """
        return cls.left_jacobian_inv(tau).T


    @classmethod
    def from_axis_angle(cls, axis: Vec3, theta: float) -> SO3:
        return cls.Exp(axis.normalized()*theta)

    def inverse(self) -> SO3:
        return SO3(self._R.T)

    def rotate(self, p: Vec3) -> Vec3:
        return Vec3(*(self._R @ p.v))

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


#  SE3 representation: a rotation + translation pair — (SO3, Vec3) (or quaternion + Vec3), or a 4×4 matrix. The pair is cleaner.
#  se(3) hat/vee: ξ = (ρ, θ) ∈ R^6, hat(ξ) = [[Φ, ρ], [0, 0]].
#  SE3.Exp(ξ): rotation R = SO3.Exp(θ), translation t = V ρ where V = left_jacobian(θ) — you already have it.
#  SE3.Log(T): θ = R.Log(), ρ = V^{-1} t where V^{-1} = left_jacobian_inv(θ) — also already have it.
#  Round-trip test Log(Exp(ξ)) == ξ, same shape as your SO(3) one.


class SE3Tangent:
    """ SE3Tangent represents an element of se(3), the Lie algebra of SE(3).

        While se(3) is formally a set of 4x4 matrices hat(xi); 
        here we store the element by its R^6 coordinates xi = vee(hat(xi)).

            xi = (rho, theta)

        with rho : translation
             theta : rotation    (cf Fig 7.2 in Barfoot)

        Elements of se(3) are also called "twists".
    """
    def __init__(self, rho: Vec3, theta: Vec3):
        self._rho = rho
        self._theta = theta

    @property
    def rho(self) -> Vec3: return self._rho

    @property
    def theta(self) -> Vec3: return self._theta

    def to_numpy(self) -> np.ndarray:
        """ Returns an element of tangent space se(3) as a (6,) numpy array
        """
        return np.concatenate([self._rho.to_numpy(), self._theta.to_numpy()])

    @classmethod
    def from_numpy(cls, xi: np.ndarray) -> SE3Tangent:
        """ Constucts an element of se(3) from a (6,) numpy array

            Assumes structure xi = (rho, theta)
            with   rho : translation
                 theta : rotation    (cf Fig 7.2 in Barfoot)
        """
        rho = Vec3(*xi[:3])
        theta = Vec3(*xi[3:])
        return cls(rho, theta)

    def hat(self) -> np.ndarray:
        """se(3) element xi^ (4x4) of this tangent:

            xi^ = [[ [theta]_x,  rho ],
                   [   0  0  0,    0  ]]

            Used just for a test: Exp closed form compared to matrice exponential

        TODO HARMONIZE):
            - for SE3, hat is a method SE3Tangent
            - for SO3, hat is a static method of SO3
        """
        M = np.zeros((4, 4))
        M[:3, :3] = SO3.hat(self._theta)
        M[:3, 3]  = self._rho.to_numpy()
        return M



class SE3:
    """ A rigid motion in 3D: an element of SE(3)

        Stored as a (rotation,translation) pair (SO3, Vec3).

        Acts on points by  p -> R p + t.
    """
    def __init__(self, R: SO3, t: Vec3):
        self._R = R
        self._t = t

    def to_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3,:3] = self._R.to_matrix()
        T[:3,3] = self._t.to_numpy()  # (3,) assigned into (3,1) column
        return T

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> SE3:
        """ Constructs an element of SE(3) from a 4x4 numpy matrix

            Assumes structure  T = [R,t;0,1]
            with R : 3x3 rotation
                 t : (3,) translation
        """
        return cls(SO3(T[:3, :3]), Vec3(*T[:3, 3]))

    def __mul__(self, other: SE3) -> SE3:
        # [[R1,t1],[0,1]] @ [[R2,t2],[0,1]] = [[R1R2, R1 t2 + t1],[0,1]]
        return SE3(self._R * other._R,
                   self._R.rotate(other._t) + self._t)

    def inverse(self) -> SE3:
        Rinv = self._R.inverse()
        t = Vec3(*(-Rinv.to_matrix() @ self._t.to_numpy()))
        return SE3(Rinv, t)

    @classmethod   
        #TODO I DONT LIKE EXP on not R6
    def Exp(cls, tau: SE3Tangent) -> SE3:
        """ Exponential map  so(3)(R^6) -> SE(3)

            Takes tau = theta * u (unit axis u, angle theta = ||tau||) and returns
            the rotation by theta around u:

                R = Exp(tau) = exp(hat(tau))

            with hat(tau) the skew-symmetric matrix, element of so(3), the Lie algebra of SO(3)

            Closed form via (rodrigues, combined with screw integral):

                [exp(tau^) , V*rho
                       0,        1]

                with exp(tau^) rodrigues formula
                     V*rho the product of the SO(3) left jacobian and the translation part of the twist 

            Inverse of Log
        """
        R = SO3.Exp(tau.theta)
        V = SO3.left_jacobian(tau.theta)
        t = V @ tau.rho.to_numpy()
        return cls(R, Vec3(*t))


    def Log(self) -> SE3Tangent:
        """ Logarithmic map SE3 -> SE3Tangent (i.e. R^6)
                         SE3 to twist vector?

            Returns the twist vector

              xi = [ rho,    =   [ V^{-1} t,
                     tau ]         Log(R) ]
        """
        theta = self._R.Log()
        Vinv = SO3.left_jacobian_inv(theta)
        rho = Vec3(*(Vinv @ self._t.to_numpy()))
        return SE3Tangent(rho, theta)





