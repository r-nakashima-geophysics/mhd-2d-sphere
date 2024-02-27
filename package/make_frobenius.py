"""Makes the Frobenius solutions of 2D MHD waves on a rotating sphere
under the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

References
-----
[1] Nakashima & Yoshida (submitted)

"""

import math

import numpy as np


def calc_frobenius(m_order: int,
                   alpha: float,
                   num_theta: int,
                   eig: float,
                   mu_c: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates some coefficients of the Frobenius solutions for
    the non-Malkus field B_phi = B_0 sin(theta) cos(theta)

    Parameters
    -----
    m_order : int
        The zonal wavenumber (order)
    alpha : float
        The Lehnert number
    num_theta : int
        The number of a grid in the theta direction
    eig : float
        An eigenvalue
    mu_c : float
        A critical latitude

    Returns
    -----
    psi1 : ndarray
        The first Frobenius solution
    psi2 : ndarray
        The second Frobenius solution

    Notes
    -----
    This function is based on eqs. (17) and (18) in Nakashima & Yoshida
    (submitted)[1]_.

    """

    bc2_d: list[float] = [float(), ] * 4
    bc2_d[0] = mu_c ** 2
    bc2_d[1] = 2 * mu_c
    bc2_d[2] = 2
    bc2_d[3] = 0

    sin2: float = 1 - (mu_c**2)
    denominator: float = bc2_d[1] * sin2

    const_d: list[float] = [float(), ] * 4
    const_d[0] = -(eig/(m_order*(alpha**2))+bc2_d[1]*mu_c+2*bc2_d[0]) \
        / denominator
    const_d[1] = (2*bc2_d[1]*mu_c-bc2_d[2]*sin2/2) / denominator
    const_d[2] \
        = ((m_order**2)*bc2_d[1]/sin2-bc2_d[2]*mu_c-3*bc2_d[1]) \
        / denominator
    const_d[3] = (bc2_d[1]+bc2_d[2]*mu_c-bc2_d[3]*sin2/6) / denominator

    coeff_a: list[float] = [float(), ] * 4
    coeff_b: list[float] = [float(), ] * 4
    coeff_a[0] = const_d[0]
    coeff_a[1] \
        = ((const_d[0]**2)+2*const_d[0]*const_d[1]+const_d[2]) / 4
    coeff_b[0] = -2*const_d[0] + const_d[1]
    coeff_b[1] = (-3*(const_d[0]**2)-2*const_d[0]*const_d[1] +
                  2*(const_d[1]**2)-const_d[2]+2*const_d[3]) / 4

    lin_theta: np.ndarray = np.linspace(0, math.pi, num_theta)
    delta_mu: np.ndarray = np.cos(lin_theta) - np.full(num_theta, mu_c)

    psi1 = 1 + coeff_a[0]*delta_mu + coeff_a[1]*(delta_mu**2)
    psi2 = psi1*np.log(delta_mu) \
        + coeff_b[0]*delta_mu + coeff_b[1]*(delta_mu**2)

    return psi1, psi2
#
