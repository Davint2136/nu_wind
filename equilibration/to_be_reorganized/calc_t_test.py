import numpy as np
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from scipy.optimize import bisect

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))

# TODO: Add some exception handling for invalid nb, ye?
def temperature_from_e(table : Table, nb : np.ndarray, ye : np.ndarray, e_val : float):
    """
        Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

        Inputs:
            table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
            nb [np.NDArray]: A 1D array consisting of one baryon number density in cm^-3.
            ye [np.NDArray]: A 1D array consisting of one charge fraction.
            e_val [float]: The fluid's energy density in MeV/fm^3.

        Outputs:
            t [float]: The fluid temperature in MeV.
    """

    log_e = np.log((table.thermo["Q7"] + 1) * table.mn * nb[0] * 1e-39)
    # Find difference between calculated e and actual e
    def f(t):
        e_interp = table.eval_given_rtx(log_e, nb * 1e-39, ye, np.array([t]), method="linear")[0]
        return np.log(e_val * 1e-39) - e_interp

    # Use a bisection method to find root of f. 
    try:
        t = bisect(f, table.t[0], table.t[-1], disp=True)
    except RuntimeError:
        print("temperature_from_e could not converge to a temperature.")
    
    return t

md = Metadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads = {
        999: ("N", "average nucleous")
    },
    micro={
        10041: ("mn_d", "neutron dirac effective mass divided by neutron mass"),
        11041: ("mp_d", "proton dirac effective mass divided by proton mass")
    }
)
eos = Table(md)
print("reading...")
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

#Test using an actual gridpoint on the table, good result (difference between actual vs calculated T is O(1e-10))

nb   = 4.208366627847035e+38  # Baryon number density [cm-3]           
t = 12.406403541564941     # Temperature [MeV]
ye   = 0.07158458232879639    # Electron fraction
print(nb * 1e-39 * eos.unit_dens)

inb = np.argmin(np.abs(eos.nb - (nb * 1e-39)))
iye = np.argmin(np.abs(eos.yq - ye))
it = np.argmin(np.abs(eos.t - t))

log_e = np.log((eos.thermo["Q7"] + 1) * eos.mn * nb * 1e-39)
e_val = np.exp(eos.eval_given_rtx(log_e, np.array([nb]) * 1e-39, np.array([ye]), np.array([t]))[0]) * 1e39
print(e_val)

temp = temperature_from_e(eos, np.array([nb]), np.array([ye]), e_val)
print(f"Calculated temperature: {temp}")
print(f"Actual temperature: {t}")
print(f"Temperature error: {np.abs(t - temp)}")

"""
log_e = np.log((eos.thermo["Q7"] + 1) * eos.mn * nb * 1e-39)
e_val = np.exp(eos.eval_given_rtx(log_e, np.array([nb]) * 1e-39, np.array([ye]), np.array([t])))[0] * 1e39
print(e_val)
temp = temperature_from_e(eos, np.array([nb]) * 1e-39, np.array([ye]), e_val *1e-39)
"""