import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from os import system, name
from scipy.optimize import bisect
import pandas as pd

#TODO: Need to figure out how to start the simulation with a given n_m1, J_m1 and then create a state object with the distributions, using that as the initial state.
class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1, time = 0.0, dU = None):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV cm-3]
        self.n_m1 = n_m1                                    # Gray neutrino number densities. Order: nue, anue, nux, anux [cm-3]
        self.J_m1 = J_m1                                    # Gray neutrino energy densities. Order: nue, anue, nux, anux [MeV cm-3]
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.edot_sources = None                                 # Neutrino energy density source terms for integration [MeV cm-3 s-1]
        self.ndot_sources = None                                 # Neutrino number density source terms for integration [cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None                                  # Dirac effective proton mass [MeV]
        self.mn_eff = None                                  # Dirac effective neutron mass [MeV]
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = dU
        self.time = time                                    # Time at which the state occurs during the simulation [s]


class Solver:
    c = 29979245800.0

    def __init__(self, table : Table, state : State, opacity_flags : dict, opacity_pars : dict, binwidth : float):
        self.timestep = None
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        self.E_range = np.arange(2.5, 100, binwidth)
        """self.k1_e = []
        self.k2_e = []
        self.k1_ye = []
        self.k2_ye = []"""


    def temperature_from_e(self, state : State):
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
                nb [np.NDArray]: A 1D array consisting of one baryon number density in cm^-3.
                ye [np.NDArray]: A 1D array consisting of one charge fraction.
                e_val [float]: The fluid's energy density in MeV/cm^3.

            Outputs:
                t [float]: The fluid temperature in MeV.
        """

        log_e = np.log((self.table.thermo["Q7"] + 1) * self.table.mn * state.nb * 1e-39)
        # Find difference between calculated e and actual e
        def f(t):
            e_interp = self.table.eval_given_rtx(log_e, np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([t]), method="linear")[0]
            return (state.fluid_e * 1e-39) - np.exp(e_interp)

        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")
        
        return t
    
    def get_potentials(self, state : State):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """


        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn 
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q  #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self, state : State):
        """
            Calulates quantities needed for BNS_NURATES corrections that are relevant at high density. 

            Outputs:
                mp_eff [float]: Proton Dirac effective mass [MeV]
                pn_eff [float]: Neutron Dirac effective mass [MeV]
                dm_eff [float]: The effective mass difference between neutrons and protons
        """

        #Use eos.qK to find dirac effective masses to calculate dm_eff, don't attempt dU for now

        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff
        return mp_eff, mn_eff, dm_eff

    def calculate_distribution(self, state: State):
        #Initialize eos_pars and populate
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = state.nb * 1e-21 # Convert baryon number density to nm^-3
        eos_pars.temp = state.t
        eos_pars.ye = state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = state.mu_n
        eos_pars.mu_e = state.mu_e
        eos_pars.mu_p = state.mu_p
        eos_pars.dm_eff = state.dm_eff

        #Create a quadrature, populate it with data for a Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 6
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars
        
        #Load M1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = state.chi_m1
        m1_pars.n = [x * 1e-21 for x in state.n_m1]
        m1_pars.J = [x * 1e-21 for x in state.J_m1]

        distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)

        #Initialize energy distribution - from 0 to 100 MeV in 5 MeV steps. Center of bin used for computation.
        bin_width = 5 #MeV
        nu_E_bins = np.arange(2.5, 100, bin_width)
        
        #Find per-species neutrino distribution
        ids = {0 : "nue", 1 : "anue", 2 : "nux", 3: "anux"}
        nu_dist = {}

        for idx in range(0, 4):
            species_dist = []
            for E in nu_E_bins:
                f_point = bns.TotalNuF(E, distr_pars, idx)
                species_dist.append(f_point)
            nu_dist[ids[idx]] = np.array(species_dist)

        #Find nu_n(E) to get neutrino number densities at the mid-bin points
        nu_n_dist = {"nue" : nu_dist["nue"] * state.n_m1, "anue" : nu_dist["anue"] * state.n_m1, "nux" : state.n_m1, "anux" : state.n_m1}
        return nu_n_dist
        

    

    #TODO: do this using spectral rates. Make bins for different neutrino energies, find the rates for each.
    def calculate_spectral_rates(self, state : State):
        """
            Calculates neutrino interaction rates based on the fluid's conditions in the current state.

            Outputs:
                gray_rates [dict]: A dictionary consisting of energy and number emissivities, energy and number absorptivities, and scattering absorptivities.
        """

        return 



#TESTING-------------------------------------------------------------
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

## Input thermodynamic quantities
##  N.B.: chemical potentials include the rest mass contribution

nb = (9.81e13 / (eos.unit_dens * eos.mn)) * 1e39 # Baryon number density [cm-3]   
t = 16.63   # Temperature [MeV]
ye   =  0.06

e_nb = np.exp(eos.eval_given_rtx(np.log((eos.thermo["Q7"] + 1) * nb * 1e-39 * eos.mn), np.array([nb]) * 1e-39, np.array([ye]), np.array([t]), method="linear"))[0] * 1e39

n_m1 = [0.36e-3 * nb, 2.21e-3 * nb, 0.92e-3 * nb, 0.92e-3 * nb]  # Neutrino number densities [cm-3]
J_m1 = [0.18e-1 * nb, 1.23e-1 * nb, 0.48e-1 * nb, 0.48e-1 * nb]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1/3, 1/3, 1/3, 1/3]  # Eddington factor
dU = 3.48e1


init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': False, 'use_WM_ab': False, 'use_WM_sc': False, 'use_decay': True, 'brem_implementation': 'GP19', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state, opacity_flags, opacity_pars, 5.0)

solver.states[0].t = solver.temperature_from_e(init_state)
solver.states[0].mu_p, solver.states[0].mu_n, solver.states[0].mu_e = solver.get_potentials(init_state)
solver.states[0].mp_eff, solver.states[0].mn_eff, solver.states[0].dm_eff = solver.calculate_corrector_quantities(init_state)
solver.calculate_spectral_rates(init_state)
