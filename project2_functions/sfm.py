import numpy as np


def skew(a):
    """Skew-symmetric matrix such that skew(a) @ v == np.cross(a, v)."""
    a = np.asarray(a).ravel()
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])


def compute_fundamental(x1, x2):
    """
    Compute fundamental matrix from corresponding homogeneous points.
    x1, x2: 3 x N arrays.
    """
    n = x1.shape[1]

    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    A = np.zeros((n, 9))

    for i in range(n):
        A[i] = [
            x1[0, i] * x2[0, i],
            x1[0, i] * x2[1, i],
            x1[0, i] * x2[2, i],
            x1[1, i] * x2[0, i],
            x1[1, i] * x2[1, i],
            x1[1, i] * x2[2, i],
            x1[2, i] * x2[0, i],
            x1[2, i] * x2[1, i],
            x1[2, i] * x2[2, i],
        ]

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank 2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    return F


def compute_fundamental_normalized(x1, x2):
    """
    Normalized 8-point algorithm.
    x1, x2: 3 x N homogeneous point arrays.
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Number of points don't match.")

    x1 = x1 / x1[2]
    x2 = x2 / x2[2]

    mean1 = np.mean(x1[:2], axis=1)
    std1 = np.std(x1[:2])
    s1 = np.sqrt(2) / std1

    T1 = np.array([
        [s1, 0, -s1 * mean1[0]],
        [0, s1, -s1 * mean1[1]],
        [0, 0, 1]
    ])

    mean2 = np.mean(x2[:2], axis=1)
    std2 = np.std(x2[:2])
    s2 = np.sqrt(2) / std2

    T2 = np.array([
        [s2, 0, -s2 * mean2[0]],
        [0, s2, -s2 * mean2[1]],
        [0, 0, 1]
    ])

    x1n = T1 @ x1
    x2n = T2 @ x2

    F = compute_fundamental(x1n, x2n)

    # Denormalize
    F = T1.T @ F @ T2

    if abs(F[2, 2]) > 1e-12:
        F = F / F[2, 2]

    return F


def compute_epipole(F):
    """
    Compute right epipole from F.
    Use compute_epipole(F.T) for the left epipole.
    """
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]

    if abs(e[2]) > 1e-12:
        return e / e[2]

    return e


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    """
    Plot epipolar line F*x = 0 in an image.
    Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    m, n = im.shape[:2]
    line = F @ x

    t = np.linspace(0, n, 100)

    if abs(line[1]) < 1e-12:
        return

    lt = np.array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])
    ndx = (lt >= 0) & (lt < m)

    plt.plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)

        if abs(epipole[2]) > 1e-12:
            epipole = epipole / epipole[2]

        plt.plot(epipole[0], epipole[1], "r*")


def triangulate_point(x1, x2, P1, P2):
    """
    Triangulate one point pair using least squares.
    x1, x2: homogeneous image points.
    P1, P2: 3 x 4 camera matrices.
    """
    M = np.zeros((6, 6))

    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    _, _, Vt = np.linalg.svd(M)
    X = Vt[-1, :4]

    return X / X[3]


def triangulate(x1, x2, P1, P2):
    """
    Triangulate many corresponding points.
    x1, x2: 3 x N homogeneous point arrays.
    Returns X: 4 x N homogeneous 3D points.
    """
    n = x1.shape[1]

    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]

    return np.array(X).T


def compute_P(x, X):
    """
    Compute camera matrix from 2D-3D correspondences.
    x: 3 x N homogeneous image points.
    X: 4 x N homogeneous 3D points.
    """
    n = x.shape[1]

    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    M = np.zeros((3 * n, 12 + n))

    for i in range(n):
        M[3 * i, 0:4] = X[:, i]
        M[3 * i + 1, 4:8] = X[:, i]
        M[3 * i + 2, 8:12] = X[:, i]
        M[3 * i:3 * i + 3, i + 12] = -x[:, i]

    _, _, Vt = np.linalg.svd(M)

    return Vt[-1, :12].reshape((3, 4))


def compute_P_from_fundamental(F):
    """
    Compute second camera matrix from fundamental matrix.
    Assumes P1 = [I | 0].
    This gives a projective reconstruction.
    """
    e = compute_epipole(F.T)
    Te = skew(e)

    return np.vstack((Te @ F.T, e)).T


def compute_P_from_essential(E):
    """
    Compute four possible second camera matrices from essential matrix.
    Assumes P1 = [I | 0].
    """
    U, S, Vt = np.linalg.svd(E)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    E = U @ np.diag([1, 1, 0]) @ Vt

    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    P2 = [
        np.hstack((U @ W @ Vt, U[:, 2:3])),
        np.hstack((U @ W @ Vt, -U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, U[:, 2:3])),
        np.hstack((U @ W.T @ Vt, -U[:, 2:3])),
    ]

    return P2


def camera_center(P):
    """
    Compute camera center C from camera matrix P.
    P = [M | p4], C = -M^{-1} p4.
    """
    M = P[:, :3]
    p4 = P[:, 3]

    return -np.linalg.inv(M) @ p4


class RansacModel:
    """
    RANSAC model for fundamental matrix estimation.
    Compatible with old-style ransac.py implementations.
    """

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """
        Estimate F from 8 correspondences.
        data shape expected: N x 6.
        """
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        return compute_fundamental_normalized(x1, x2)

    def get_error(self, data, F):
        """
        Compute Sampson distance for correspondences.
        data shape expected: N x 6.
        """
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        Fx1 = F @ x1
        Fx2 = F.T @ x2

        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2

        err = np.diag(x1.T @ F @ x2) ** 2 / denom

        return err


def F_from_ransac(x1, x2, model=None, maxiter=5000, match_threshold=1e-6):
    """
    Robust estimation of F using OpenCV RANSAC if available.
    x1, x2: 3 x N homogeneous points.

    Returns:
        F, inliers
    """
    try:
        import cv2

        pts1 = (x1[:2] / x1[2]).T.astype(np.float32)
        pts2 = (x2[:2] / x2[2]).T.astype(np.float32)

        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99
        )

        if F is None:
            raise RuntimeError("OpenCV could not estimate F.")

        inliers = np.where(mask.ravel() == 1)[0]

        return F, inliers

    except ImportError:
        raise ImportError(
            "OpenCV is not installed. Install it with:\n"
            "pip install opencv-python\n"
            "or implement/use the old ransac.py from the book."
        )