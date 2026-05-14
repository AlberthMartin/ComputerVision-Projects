import numpy as np

import numpy as np
from scipy import linalg


class Camera(object):
    """Class for representing pin-hole cameras.
    """
    def __init__(self, P):
        self.P = P
        # Calibration matrix
        self.K = None
        # Rotation
        self.R = None
        # Translation
        self.t = None
        # Camera center
        self.c = None

    def project(self, X):
        """Project points in X (4 * n array)

           and normalize coordinates.
        """
        x = np.dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

    def factor(self):
        """Factorize the camera matrix into K, R, t as P = K[R / t].
        """

        # Factor first 3x3 part
        K, R = linalg.rq(self.P[:, :3])

        # Make diagonal of K positive
        T = np.diag(np.sign(np.diag(K)))
        if linalg.det(T) > 0:
            T[1, 1] *= -1

        self.K = np.dot(K, T)
        self.R = np.dot(T, R)
        self.t = np.dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def center(self):
        """Compute and return the camera center.
        """
        if self.c is not None:
            return self.c
        else:
            # Compute c by factoring
            self.factor()
            self.c = -np.dot(self.R.T, self.t)
            return self.c


def rotation_matrix(a):
    """Creates a 3d rotation matrix for a rotation

       around the axis of a vector a.
    """
    R = np.eye(4)
    R[:3, :3] = linalg.expm([[0, -a[2], a[1]],
                             [a[2], 0, -a[0]],
                             [-a[1], a[0], 0]])
    return R


