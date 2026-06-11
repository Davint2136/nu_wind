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


    def compute_t(self):
        
        pass
    
    def get_potentials(self):
        pass

    def calculate_corrector_quantities(self):
        pass

    def calculate_rates(self):
        pass

    def calculate_source_terms(self):
        pass

    def integrate_step(self):
        pass


