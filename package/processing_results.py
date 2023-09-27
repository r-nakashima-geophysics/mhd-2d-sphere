"""Processes obtained data"""

import math

import numpy as np


def screening_eig_q(criterion_q: float,
                    bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray,
                 np.ndarray, np.ndarray, np.ndarray]:
    """Check the Q-values of eigenmodes

    Parameters
    -----
    criterion_q : float
        A criterion for plotted eigenvalues based on the Q value
    bundle : tuple of ndarray
        A tuple of results

    Returns
    -----
    bundle : tuple of ndarray
        A tuple of results

    """

    eig: np.ndarray
    mke: np.ndarray
    mme: np.ndarray
    ohm: np.ndarray
    sym: np.ndarray
    eig, mke, mme, ohm, sym \
        = bundle[1], bundle[2], bundle[3], bundle[4], bundle[5]

    size_mat: int = eig.shape[1]
    num_alpha: int = extract_param(bundle)[2]

    check_q: np.ndarray

    for i_alpha in range(num_alpha):

        check_q = np.abs(eig[i_alpha, :].real) \
            + 2*eig[i_alpha, :].imag*criterion_q

        for i_mode in range(size_mat):
            if check_q[i_mode] <= 0:
                eig[i_alpha, i_mode] = math.nan
                mke[i_alpha, i_mode] = math.nan
                mme[i_alpha, i_mode] = math.nan
                ohm[i_alpha, i_mode] = math.nan
                sym[i_alpha, i_mode] = None
            #
        #
    #

    bundle = (bundle[0], eig, mke, mme, ohm, sym)

    return bundle
#


def extract_param(bundle: tuple[np.ndarray, np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray, np.ndarray]) \
        -> tuple[float, float, int, float]:
    """Extracts some parameters from results

    Parameters
    -----
    bundle : tuple of ndarray
        A tuple of results

    Returns
    -----
    alpha_init : float
        The initial value of alpha
    alpha_end : float
        The end value of alpha
    num_alpha : int
        The number of alpha
    ohm_max : float
        The maximum value of ohmic dissipation

    """

    lin_alpha: np.ndarray = bundle[0]
    ohm: np.ndarray = bundle[4]

    alpha_init: float = lin_alpha[0]
    alpha_end: float = lin_alpha[-1]

    num_alpha: int = len(lin_alpha)

    ohm_max: float
    if ohm:
        ohm_max = np.nanmax(ohm)
    else:
        ohm_max = 0
    #

    return alpha_init, alpha_end, num_alpha, ohm_max
#


def sort_sv(sym_alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorts into sinuous and varicose modes

    Parameters
    -----
    sym_alpha : ndarray
        The symmetry of eigenmodes for a given alpha

    Returns
    -----
    sinuous : ndarray
        The trace of sinuous modes
    varicose : ndarray
        The trace of varicose modes

    """

    size_mat: int = sym_alpha.shape[0]

    sinuous = np.full(size_mat, np.nan)
    varicose = np.full(size_mat, np.nan)
    for i_mode in range(size_mat):
        if sym_alpha[i_mode] == 'sinuous':
            sinuous[i_mode] = 1
        elif sym_alpha[i_mode] == 'varicose':
            varicose[i_mode] = 1
        #
    #

    return sinuous, varicose
#


def sort_pr(eig_alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorts into prograde and retrograde modes

    Parameters
    -----
    eig_alpha : ndarray
        Eigenvalues for a given alpha

    Returns
    -----
    prograde : ndarray
        The trace of prograde modes
    retrograde : ndarray
        The trace of retrograde modes

    """

    size_mat: int = eig_alpha.shape[0]

    prograde: np.ndarray = np.full(size_mat, np.nan)
    retrograde: np.ndarray = np.full(size_mat, np.nan)
    for i_mode in range(size_mat):
        if eig_alpha[i_mode].real > 0:
            prograde[i_mode] = 1
        elif eig_alpha[i_mode].real < 0:
            retrograde[i_mode] = 1
        #
    #

    return prograde, retrograde
#


def pickup_unstable(eig_alpha: np.ndarray) -> np.ndarray:
    """Picks up unstable modes

    Parameters
    -----
    eig_alpha : ndarray
        Eigenvalues for a given alpha

    Returns
    -----
    unstable : ndarray
        The trace of unstable modes

    """

    size_mat: int = eig_alpha.shape[0]

    unstable: np.ndarray = np.full(size_mat, np.nan)
    for i_mode in range(size_mat):
        if math.fabs(eig_alpha[i_mode].imag) > 0:
            unstable[i_mode] = 1
        #
    #

    return unstable
#


def sort_alfvenic(mke_alpha: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Sorts into alfvenic and non-alfvenic modes

    Parameters
    -----
    mke_alpha : ndarray
        The mean kinetic energy for a given alpha

    Returns
    -----
    alfvenic : ndarray
        The trace of alfvenic modes
    non_alfvenic : ndarray
        The trace of non-alfvenic modes

    """

    size_mat: int = mke_alpha.shape[0]

    alfvenic: np.ndarray = np.full(size_mat, np.nan)
    non_alfvenic: np.ndarray = np.full(size_mat, np.nan)
    for i_mode in range(size_mat):
        if mke_alpha[i_mode] > 0.51:
            non_alfvenic[i_mode] = 1
        elif mke_alpha[i_mode] < 0.49:
            non_alfvenic[i_mode] = 1
        else:
            alfvenic[i_mode] = 1
    #

    return alfvenic, non_alfvenic
#
