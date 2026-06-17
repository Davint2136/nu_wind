import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy


class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV]
        self.n_m1 = n_m1                                    # Gray neutrino number densities. Order: nue, anue, nux, anux [cm-3]
        self.J_m1 = J_m1                                    # Gray neutrino energy densities. Order: nue, anue, nux, anux [MeV cm-3]
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.potentials = None                              # Chemical potentials
        self.source_terms = {"edot" : None, "ndot" : None}  # Neutrino transport source terms [MeV cm-3 s-1, cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.dm_eff = 0                                     # Nucleon effective mass difference [MeV]
        self.dU = 0                                         # Nucleon interaction potential difference [MeV]


class Solver:
    def __init__(self, table, state, integrate_tolerance, opacity_flags, opacity_params):
        self.timestep = None
        self.integrate_tolerance = integrate_tolerance
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_params = opacity_params


    def temperature_from_e(self, nb : np.ndarray, ye : np.ndarray, e_val : float):
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
        from scipy.optimize import bisect

        new_nb = np.array([nb])
        new_ye = np.array([ye])

        # Find difference between calculated e and actual e
        def f(t):
            interp = self.table.interpolate(nb, ye, np.array([t]), method='linear')
            e_table = (interp.thermo["Q7"] + 1) * interp.mn * nb
            return e_val - e_table[0, 0, 0]

        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")
        
        return t
    
    def get_potentials(self):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """
        interp = self.table.interpolate3D(self.states[-1].nb, self.states[-1].ye, self.states[-1].t, method='linear')
        mu_b = (self.table.thermo["Q3"][0, 0, 0] + 1) * self.table.mn
        mu_q = self.table.thermo["Q4"][0, 0, 0] * self.table.mn
        mu_l = self.table.thermo["Q5"][0, 0, 0] * self.table.mn

        mu_p = mu_b + mu_q
        mu_n = mu_b
        mu_e = mu_l - mu_q

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self):
        # TODO: Use eos.micro to find dirac effective masses to calculate dm_eff, don't attempt dU for now
        pass

    def calculate_rates(self):
        # TODO: Extremely similar to bns_nurates' test_bindings.py. Use the same code structure
        pass

    def calculate_source_terms(self):
        pass

    def integrate_step(self):
        # TODO: Build and test RK2 integrator before implementing anything here. Best to do this in a separate file.
        pass


