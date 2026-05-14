import numpy as np
from scipy.ndimage import filters


def plane_sweep_ssd(im_l, im_r, start, steps, wid):
    """
    Find disparity image using sum of squared differences (SSD).

    Parameters
    ----------
    im_l : ndarray
        Left grayscale image.
    im_r : ndarray
        Right grayscale image.
    start : int
        Starting disparity.
    steps : int
        Number of disparity values to test.
    wid : int
        Patch/window size used by uniform filtering.

    Returns
    -------
    disparity : ndarray
        Disparity map. Each pixel stores the best disparity index.
    """

    im_l = im_l.astype(np.float64)
    im_r = im_r.astype(np.float64)

    m, n = im_l.shape

    # Store SSD costs for every disparity
    dmaps = np.zeros((m, n, steps))

    for displ in range(steps):
        # Same shifting style as Solem's NCC code
        shifted_l = np.roll(im_l, -displ - start, axis=1)

        # Pixel-wise squared difference
        diff = (shifted_l - im_r) ** 2

        # Sum over local patch using filtering
        ssd = filters.uniform_filter(diff, wid)

        dmaps[:, :, displ] = ssd

    # SSD is a cost, so choose the disparity with minimum SSD
    disparity = np.argmin(dmaps, axis=2)

    return disparity + start


def plane_sweep_ssd_gauss(im_l, im_r, start, steps, wid):
    """
    SSD stereo using Gaussian filtering instead of uniform filtering.
    This is analogous to Solem's Gaussian NCC version.
    """

    im_l = im_l.astype(np.float64)
    im_r = im_r.astype(np.float64)

    m, n = im_l.shape
    dmaps = np.zeros((m, n, steps))

    for displ in range(steps):
        shifted_l = np.roll(im_l, -displ - start, axis=1)

        diff = (shifted_l - im_r) ** 2

        ssd = filters.gaussian_filter(diff, wid)

        dmaps[:, :, displ] = ssd

    disparity = np.argmin(dmaps, axis=2)

    return disparity + start



def rof_denoise(im, U_init=None, tolerance=0.1, tau=0.125, tv_weight=100):
    """
    Rudin-Osher-Fatemi denoising using Chambolle's projection method.

    This is adapted to work well for disparity/depth maps.

    Parameters
    ----------
    im : ndarray
        Input noisy image/disparity map.
    U_init : ndarray or None
        Initial guess. If None, uses the input image.
    tolerance : float
        Stopping threshold.
    tau : float
        Step size.
    tv_weight : float
        Denoising weight. Larger values produce stronger smoothing.

    Returns
    -------
    U : ndarray
        Denoised image.
    """

    im = im.astype(np.float64)

    m, n = im.shape

    if U_init is None:
        U = im.copy()
    else:
        U = U_init.copy()

    Px = np.zeros((m, n))
    Py = np.zeros((m, n))

    error = 1.0

    while error > tolerance:
        Uold = U.copy()

        # Gradient of primal variable
        GradUx = np.roll(U, -1, axis=1) - U
        GradUy = np.roll(U, -1, axis=0) - U

        # Update dual variable
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy

        NormNew = np.maximum(1, np.sqrt(PxNew ** 2 + PyNew ** 2))

        Px = PxNew / NormNew
        Py = PyNew / NormNew

        # Divergence
        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)

        DivP = (Px - RxPx) + (Py - RyPy)

        # Update primal variable
        U = im + tv_weight * DivP

        error = np.linalg.norm(U - Uold) / np.sqrt(n * m)

    return U