import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))


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
    }
)
eos = Table(md)
print("Reading...")
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

"""
Method to find t, given nb, ye, and e:

1. find energy per baryon by dividing e by nb. This should by eos.thermo["Q7"] after undoing its scaling.
2. Loop through the density + electron fraction slice of eos.thermo["Q7"] by temperature until the two closest e values are found (higher and lower)
3. Interpolate the table according to the fixed nb and ye. For the temperature, use an array of points between the two temperatures found in step 2
4. Loop through the table by temperature until the t corresponding to the closest e to the input e is found.
5. The closest t is the temperature
"""

#Simple test. Works, but eos.t doesn't have the temperature resolution to get the exact temperature.
#Only gets as close as the closest temperature in eos.t

def calculate_t(nb, yq, e, table):
    
    #Get slice of table for that specific nb and yq
    print("Interpolating...")
    nb_ye_slice = eos.interpolate(np.array([nb]), np.array([yq]), method='linear') #effectively creates a 1D table
    e_table = (nb_ye_slice.thermo["Q7"] + 1) * nb_ye_slice.mn * nb
    temp = nb_ye_slice.t[np.argmin(np.abs(e_table - e))]
    return temp

# Read table from file
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
    }
)
eos = Table(md)
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

#Generate test point using point F of chiesa et. al. 2025
nb = 4.208366627847035e+38 * 1e-39 #fm^-3
ye = 0.07
t = 12.39 # MeV
test_table = eos.interpolate(np.array([nb]), np.array([0.07]), np.array([t]), method='linear')
fluid_e = (test_table.thermo["Q7"][0, 0, 0] + 1) * test_table.mn * test_table.nb[0]

print(nb, ye, t, fluid_e)
print(calculate_t(nb, ye, fluid_e, eos)) #works! Kind of...

#PRIMITIVE SOLVER PYTHON PORT----------------------------------------------------
"""
Function Steps:

1. Input the table, value of the table variable, nb, and yq
2. Finds the closest index corresponding to values less than nb and yq
3. Calculate weights to describe distance of selected grid point to input
4. Use bisection method strategy found in primitive_solver to minimize difference
   between var from table lookup and input var
5. Output corresponding temperature once minimized via interpolation

Pros:
    1. Only one interpolation via Scipy occurs, function f from primitive-solver is fast
    2. Bisection method is reasonably fast for ensuring var_pt converges to var (linear), could switch to a secant method
       if speed becomes a concern. Newton's method is off the table (derivative calculation would be slow and innacurate)
    3. porting primitive-solver's f function will allow much greater accuracy due to exact weights

Cons:
    1. Python is slow as a whole, but it shouldn't be too big of an issue
    2. Bisection method's termination condition (i.e. tolerance) can affect accuracy, try not to be too precise
"""



