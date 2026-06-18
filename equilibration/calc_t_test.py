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
            nb [np.NDArray]: A 1D array consisting of one baryon number density in fm^-3.
            ye [np.NDArray]: A 1D array consisting of one charge fraction.
            e_val [float]: The fluid's energy density in MeV/fm^3.

        Outputs:
            t [float]: The fluid temperature in MeV.
    """

    # Find difference between calculated e and actual e
    def f(t):
        interp = table.interpolate(nb, ye, np.array([t]), method='linear')
        e_table = (interp.thermo["Q7"] + 1) * interp.mn * nb
        return e_val - e_table[0, 0, 0]

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

#Generate test point using point F of chiesa et. al. 2025
nb = 4.208366627847035e+38 #cm^-3
ye = 0.07158458232879639
t = 12.406403541564941 # MeV
test_table = eos.interpolate(np.array([nb]) * 1e-39, np.array([0.07]), np.array([t]), method='linear')
fluid_e = (test_table.thermo["Q7"][0, 0, 0] + 1) * test_table.mn * test_table.nb[0] * 1e39
print(fluid_e)
temp = temperature_from_e(eos, np.array([nb]) * 1e-39, np.array([ye]), fluid_e * 1e-39)
print(temp)
print(f"Temperature error: {np.abs(t - temp)}")