import numpy as np
from scipy.ndimage import filters

"""
For each possible disparity:
    shift one image
    compare it with the other image
    compute SSD cost
    Choose the disparity with the lowest SSD cost for every pixel
"""
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

    # Store SSD costs for every pixel and every tested disparity.
    dmaps = np.zeros((m, n, steps))

    # Loop over all disparity values.
    for displ in range(steps):
        # Shift the left image according to the current disparity.
        shifted_l = np.roll(im_l, -displ - start, axis=1)

        # Compute pixel-wise squared difference.
        diff = (shifted_l - im_r) ** 2

        # Smooth the squared differences using a Gaussian filter.
        #
        # Unlike uniform_filter, Gaussian filtering weights the center
        # pixels more strongly than pixels near the edge of the window.
        ssd = filters.gaussian_filter(diff, wid)

        # Store the cost map for this disparity.
        dmaps[:, :, displ] = ssd

    # Choose the disparity with the lowest SSD cost for each pixel.
    disparity = np.argmin(dmaps, axis=2)

    # Convert disparity index to actual disparity value.
    return disparity + start



def rof_denoise(im, U_init=None, tolerance=0.1, tau=0.125, tv_weight=100):
    """
    Rudin-Osher-Fatemi denoising using Chambolle's projection method.

    ROF denoising is based on total variation smoothing.
    It is useful for disparity/depth maps because it reduces noise
    while trying to preserve edges.

    Parameters
    ----------
    im : ndarray
        Input noisy image or disparity map.

    U_init : ndarray or None
        Initial guess for the denoised result.
        If None, the original image is used.

    tolerance : float
        Stopping threshold.
        The iteration stops when the change between iterations becomes small.

    tau : float
        Step size used in the iterative update.

    tv_weight : float
        Denoising weight.
        Larger values produce stronger smoothing.

    Returns
    -------
    U : ndarray
        Denoised image/disparity map.
    """

    # Convert input to floating point for numerical calculations.
    im = im.astype(np.float64)

    # Get image dimensions.
    m, n = im.shape

    # Initialize U, the denoised image.
    #
    # If no initial estimate is given, start from the noisy input image.
    if U_init is None:
        U = im.copy()
    else:
        U = U_init.copy()

    # Px and Py are the dual variables used by Chambolle's method.
    #
    # You can think of them as variables that help control the amount
    # of smoothing in the x and y directions.
    Px = np.zeros((m, n))
    Py = np.zeros((m, n))

    # Initial error is set high so the while loop starts.
    error = 1.0

    # Repeat until the solution stops changing significantly.
    while error > tolerance:

        # Save previous result so we can measure how much U changes.
        Uold = U.copy()

        # Compute forward differences, which approximate image gradients.
        #
        # GradUx measures horizontal changes:
        # large value means a strong change from one pixel to the next in x.
        #
        # GradUy measures vertical changes:
        # large value means a strong change from one pixel to the next in y.
        GradUx = np.roll(U, -1, axis=1) - U
        GradUy = np.roll(U, -1, axis=0) - U

        # Update the dual variables using the gradient of U.
        #
        # tau controls the step size.
        # tv_weight controls the strength of the denoising.
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy

        # Normalize the dual variables.
        #
        # This prevents the update from becoming too large and enforces
        # the total variation constraint.
        NormNew = np.maximum(
            1,
            np.sqrt(PxNew ** 2 + PyNew ** 2)
        )

        Px = PxNew / NormNew
        Py = PyNew / NormNew

        # Compute backward-shifted versions of Px and Py.
        #
        # These are used to compute the divergence of the dual field.
        RxPx = np.roll(Px, 1, axis=1)
        RyPy = np.roll(Py, 1, axis=0)

        # Compute divergence.
        #
        # Divergence is roughly the opposite of gradient.
        # It tells how the smoothing field flows into or out of each pixel.
        DivP = (Px - RxPx) + (Py - RyPy)

        # Update the denoised image.
        #
        # U stays close to the original image im,
        # but is modified by the total variation smoothing term.
        U = im + tv_weight * DivP

        # Compute how much U changed from the previous iteration.
        #
        # If the change is smaller than tolerance, stop iterating.
        error = np.linalg.norm(U - Uold) / np.sqrt(n * m)

    # Return the final denoised image.
    return U