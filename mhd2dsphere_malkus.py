"""A code for 2D ideal MHD waves on a rotating sphere under the Malkus
field B_phi = B_0 sin(theta)

Plots 1-3 figures (linear-linear, energy partitioning, and log-log)
concerning the dispersion relation for 2D ideal MHD waves on a rotating
sphere under the Malkus field B_phi = B_0 sin(theta).

Parameters
-----
M_ORDER : int
    The zonal wavenumber (order)

Raises
-----
No plotted figures
    If all of the boolean values to switch whether to plot figures are
    False.

Notes
-----
Parameters other than command line arguments are described below.

References
-----
[1] Nakashima & Yoshida (submitted)

Examples
-----
In the below example, M_ORDER will be set to the default value.
    python3 mhd2dsphere_malkus.py
In the below example, M_ORDER will be set to 2.
    python3 mhd2dsphere_malkus.py 2

"""

import logging
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np

from package.input_arg import input_m

# ========== Parameters ==========

# Boolean values to switch whether to plot figures
# 0: dispersion relation (linear-linear)
# 1: energy partitioning
# 2: dispersion relation (log-log)
SWITCH_PLOT: Final[tuple[bool, bool, bool]] = (True, True, True)

# The zonal wavenumber (order)
M_ORDER: Final[int] = input_m(1)

# Degrees
N_INIT: Final[int] = M_ORDER  # M_ORDER <= N_INIT
N_STEP: Final[int] = 1
N_END: Final[int] = 10

# The range of the Lehnert number
# linear
ALPHA_INIT: Final[float] = 0
ALPHA_STEP: Final[float] = 0.001
ALPHA_END: Final[float] = 1
# log
ALPHA_LOG_INIT: Final[float] = -4
ALPHA_LOG_STEP: Final[float] = 0.01
ALPHA_LOG_END: Final[float] = 2

# The range of eigenvalues
# linear
EIG_INIT: Final[float] = -2
EIG_END: Final[float] = 2
# log
EIG_LOG_INIT: Final[float] = -6
EIG_LOG_END: Final[float] = 2

# The range of energy partitioning
ENERGY_INIT: Final[float] = 0
ENERGY_END: Final[float] = 1

# The paths and filenames of outputs
PATH_DIR_FIG: Final[Path] \
    = Path('.') / 'fig' / 'MHD2Dsphere_malkus'
NAME_FIG_1: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_eig.png'
NAME_FIG_2: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_ene.png'
NAME_FIG_3: Final[str] = f'MHD2Dsphere_malkus_m{M_ORDER}_eiglog.png'
FIG_DPI: Final[int] = 600

# ================================

NAMES_MODE: Final[tuple[str, str]] = ('fMR', 'sMR')
NUM_MODE: Final[int] = len(NAMES_MODE)

NUM_N: Final[int] = 1 + int((N_END-N_INIT)/N_STEP)
NUM_ALPHA: Final[int] \
    = 1 + int((ALPHA_END-ALPHA_INIT)/ALPHA_STEP)
NUM_ALPHA_LOG: Final[int] \
    = 1 + int((ALPHA_LOG_END-ALPHA_LOG_INIT)/ALPHA_LOG_STEP)

LIN_N: Final[np.ndarray] = np.linspace(N_INIT, N_END, NUM_N)
LIN_ALPHA: Final[np.ndarray] \
    = np.linspace(ALPHA_INIT, ALPHA_END, NUM_ALPHA)
LIN_ALPHA_LOG: Final[np.ndarray] \
    = np.linspace(ALPHA_LOG_INIT, ALPHA_LOG_END, NUM_ALPHA_LOG)


def wrapper_eigene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A wrapper of functions to calculate the dispersion relation and
    energy partitioning

    Returns
    -----
    eig : ndarray
        Eigenvalues (linear-linear)
    ene : ndarray
        Energy partitioning
    eig_log : ndarray
        Eigenvalues (log-log)

    See Also
    -----
    calc_eig
    calc_ene
    calc_eig

    """

    eig: np.ndarray = np.zeros((NUM_N, NUM_ALPHA, NUM_MODE))
    ene: np.ndarray = np.zeros((NUM_N, NUM_ALPHA_LOG, NUM_MODE))
    eig_log: np.ndarray = np.zeros((NUM_N, NUM_ALPHA_LOG, NUM_MODE))

    n_degree: int
    alpha: float

    for i_n in range(NUM_N):
        n_degree = LIN_N[i_n]

        if (SWITCH_PLOT[0] or SWITCH_PLOT[1]):
            for i_alpha in range(NUM_ALPHA):
                alpha = LIN_ALPHA[i_alpha]

                for name_mode in NAMES_MODE:
                    eig[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_eig(n_degree, alpha, name_mode)
                #
            #
        #

        if SWITCH_PLOT[2]:
            for i_alpha in range(NUM_ALPHA_LOG):
                alpha = 10**LIN_ALPHA_LOG[i_alpha]

                for name_mode in NAMES_MODE:
                    ene[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_ene(n_degree, alpha, name_mode)
                    eig_log[i_n, i_alpha, NAMES_MODE.index(name_mode)] \
                        = calc_eig(n_degree, alpha, name_mode)
                #
            #
        #
    #

    return eig, ene, eig_log
#


def calc_eig(n_degree: int,
             alpha: float,
             name_mode: str) -> float:
    """Calculates the dispersion relation

    Parameters
    -----
    n_degree : int
        A degree of the associated Legendre polynomial
    alpha : float
        The Lehnert number
    name_mode : str
        'fMR' or 'sMR'

    Returns
    -----
    eig : float
        An eigenvalue

    Raises
    -----
    Invalid ID
        If name_mode is neither 'fMR' nor 'sMR'.

    Notes
    -----
    If n_degree = 0, eig is set to 0. This function is based on eq. (1)
    in Nakashima & Yoshida (submitted)[1]_.

    """

    eig: float

    if name_mode not in NAMES_MODE:
        logger.error('Invalid ID')
        sys.exit()
    #

    if n_degree == 0:
        eig = 0
        return eig
    #

    nn1: int = n_degree * (n_degree+1)
    sq_rt: float = math.sqrt(1 + 4*(alpha**2)*nn1*(nn1-2))

    if name_mode == 'fMR':
        eig = (-M_ORDER-M_ORDER*sq_rt) / (2*nn1)
    elif name_mode == 'sMR':
        eig = (-M_ORDER+M_ORDER*sq_rt) / (2*nn1)
    #

    return eig
#


def calc_ene(n_degree: int,
             alpha: float,
             name_mode: str) -> float:
    """Calculates energy partitioning

    Parameters
    -----
    n_degree : int
        A degree of the associated Legendre polynomial
    alpha : float
        The Lehnert number
    name_mode : str
        'fMR' or 'sMR'

    Returns
    -----
    ene : float
        Energy partitioning

    Notes
    -----
    If n_degree = 0, ene is set to 0. If alpha = 0, ene is set to 0 for
    fast MR waves and 1 for slow MR waves. This function is based on
    the equation in the caption of Fig. 2 in Nakashima & Yoshida (in
    prep.)[1]_.

    """

    ene: float

    if n_degree == 0:
        ene = 0
        return ene
    #

    if alpha == 0:
        if name_mode == 'fMR':
            ene = 0
            return ene
        if name_mode == 'sMR':
            ene = 1
            return ene
        #
    #

    ma2: float = (M_ORDER**2) * (alpha**2)
    lambda_mr: float = calc_eig(n_degree, alpha, name_mode)
    ene = (lambda_mr**2) / ((lambda_mr**2)+ma2)

    return ene
#


def plot_eig(eig: np.ndarray) -> None:
    """Plots a figure of the dispersion relation (linear-linear)

    Parameters
    -----
    eig : ndarray
        Eigenvalues

    """

    axis: plt.Axes
    fig, axis = plt.subplots(figsize=(5, 7))

    i_n: int

    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if i_n not in (0, NUM_N-1):
            axis.plot(LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('fMR')],
                      color=[1, i_n/NUM_N, 0])
            axis.plot(LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('sMR')],
                      color=[0, i_n/NUM_N, 1])
        else:
            axis.plot(LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('fMR')],
                      color=[1, i_n/NUM_N, 0],
                      label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            axis.plot(LIN_ALPHA, eig[i_n, :, NAMES_MODE.index('sMR')],
                      color=[0, i_n/NUM_N, 1],
                      label=r'$n=$'+f' {N_INIT+i_n} slow MR')
        #
    #

    axis.grid()

    axis.set_xlim(ALPHA_INIT, ALPHA_END)
    axis.set_ylim(EIG_INIT, EIG_END)

    axis.set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axis.set_ylabel(r'$\lambda=\omega/2\Omega_0$', fontsize=16)
    axis.set_title(
        r'Dispersion relation [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}\n', color='magenta', fontsize=16)

    handle: list
    label: list
    [handle, label] = axis.get_legend_handles_labels()
    order_leg: list[int] = [3, 1, 2, 0]
    handle = [handle[i_handle] for i_handle in order_leg]
    label = [label[i_label] for i_label in order_leg]
    if M_ORDER >= 3:
        leg = axis.legend(
            handles=handle, labels=label, loc='center right',
            fontsize=14)
    else:
        leg = axis.legend(
            handles=handle, labels=label, loc='lower left',
            fontsize=14)
    #
    leg.get_frame().set_alpha(1)

    axis.tick_params(labelsize=14)
    axis.minorticks_on()

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG_1
    fig.savefig(str(path_fig), dpi=FIG_DPI)
#


def plot_ene(ene: np.ndarray) -> None:
    """Plots a figure of energy partitioning

    Parameters
    -----
    ene : ndarray
        Energy partitioning

    """

    i_n: int

    axis: plt.Axes
    fig, axis = plt.subplots(figsize=(5, 5))

    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if (M_ORDER == 1) and (i_n == 0):
            axis.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1], linewidth=3)
        #

        if i_n not in (0, NUM_N-1):
            axis.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0])
            axis.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1])
        else:
            axis.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            axis.semilogx(
                10**LIN_ALPHA_LOG, ene[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$'+f' {N_INIT+i_n} slow MR')
        #
    #

    axis.grid()

    axis.set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    axis.set_ylim(ENERGY_INIT, ENERGY_END)

    axis.set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axis.set_ylabel(
        r'$\mathrm{MKE}/(\mathrm{MKE}+\mathrm{MME})$', fontsize=16)
    axis.set_title(
        r'Energy partitioning [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}\n', color='magenta', fontsize=16)

    handle: list
    label: list
    [handle, label] = axis.get_legend_handles_labels()
    order_leg: list[int] = [2, 0, 3, 1]
    handle = [handle[i_handle] for i_handle in order_leg]
    label = [label[i_label] for i_label in order_leg]
    leg = axis.legend(
        handles=handle, labels=label,
        loc='upper right', fontsize=12)
    leg.get_frame().set_alpha(1)

    axis.tick_params(labelsize=13)
    axis.minorticks_on()

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG_2
    fig.savefig(str(path_fig), dpi=FIG_DPI)
#


def plot_eig_log(eig_log: np.ndarray) -> None:
    """Plots a figure of the dispersion relation (log-log)

    Parameters
    -----
    eig_log : ndarray
        Eigenvalues (log-log)

    """

    i_n: int

    axes: np.ndarray
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i_n_inv in range(NUM_N):
        i_n = NUM_N - 1 - i_n_inv

        if (M_ORDER == 1) and (i_n == 0):
            axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$ 1 fast MR')
            axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$ 2 slow MR')
        elif i_n not in (0, NUM_N-1):
            axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0])
            axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1])
        else:
            axes[0].loglog(
                10**LIN_ALPHA_LOG,
                -eig_log[i_n, :, NAMES_MODE.index('fMR')],
                color=[1, i_n/NUM_N, 0],
                label=r'$n=$'+f' {N_INIT+i_n} fast MR')
            axes[1].loglog(
                10**LIN_ALPHA_LOG,
                eig_log[i_n, :, NAMES_MODE.index('sMR')],
                color=[0, i_n/NUM_N, 1],
                label=r'$n=$'+f' {N_INIT+i_n} slow MR')
        #
    #

    axes[0].grid()
    axes[1].grid()

    axes[0].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    axes[1].set_xlim(10**ALPHA_LOG_INIT, 10**ALPHA_LOG_END)
    axes[1].set_ylim(10**EIG_LOG_INIT, 10**EIG_LOG_END)
    axes[0].set_ylim(10**EIG_LOG_INIT, 10**EIG_LOG_END)

    axes[0].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axes[1].set_xlabel(
        r'$|\alpha|=|B_0/2\Omega_0R_0\sqrt{\rho_0\mu_\mathrm{m}}|$',
        fontsize=16)
    axes[0].set_ylabel(r'$|\lambda|=|\omega/2\Omega_0|$', fontsize=16)
    axes[0].set_title(
        r'Retrograde ($\lambda<0$)', color='magenta', fontsize=16)
    axes[1].set_title(
        r'Prograde ($\lambda>0$)', color='magenta', fontsize=16)

    handle: list
    label: list

    [handle, label] = axes[0].get_legend_handles_labels()
    leg = axes[0].legend(
        handles=handle[::-1], labels=label[::-1],
        loc='lower right', fontsize=14)
    leg.get_frame().set_alpha(1)

    [handle, label] = axes[1].get_legend_handles_labels()
    leg = axes[1].legend(
        handles=handle[::-1], labels=label[::-1],
        loc='lower right', fontsize=14)
    leg.get_frame().set_alpha(1)

    axes[0].tick_params(labelsize=13)
    axes[1].tick_params(labelsize=13)

    fig.suptitle(
        r'Dispersion relation [$B_{0\phi}=B_0\sin\theta$] : $m=$'
        + f' {M_ORDER}', color='magenta', fontsize=16)

    fig.tight_layout()

    os.makedirs(PATH_DIR_FIG, exist_ok=True)
    path_fig: Path = PATH_DIR_FIG / NAME_FIG_3
    fig.savefig(str(path_fig), dpi=FIG_DPI)
#


if __name__ == '__main__':
    TIME_INIT: Final[float] = perf_counter()

    logging.basicConfig(level=logging.INFO)
    logger: logging.Logger = logging.getLogger(__name__)

    if True not in SWITCH_PLOT:
        logger.info('No plotted figures')
        sys.exit()
    #

    results: tuple[np.ndarray, np.ndarray, np.ndarray] \
        = wrapper_eigene()

    plt.rcParams['text.usetex'] = True

    if SWITCH_PLOT[0]:
        plot_eig(results[0])
    #
    if SWITCH_PLOT[1]:
        plot_ene(results[1])
    #
    if SWITCH_PLOT[2]:
        plot_eig_log(results[2])
    #

    TIME_ELAPSED: Final[float] = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.3f} s')

    plt.show()
#
