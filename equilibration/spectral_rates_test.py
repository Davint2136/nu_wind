import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from os import system, name
from scipy.optimize import bisect
import pandas as pd
import matplotlib.pyplot as plt

#TODO: Need to figure out how to start the simulation with a given n_m1, J_m1 and then create a state object with the distributions, using that as the initial state.
class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1, time = 0.0, dU = None):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV cm-3]
        self.n_m1 = n_m1
        self.J_m1 = J_m1
        self.nu_n_dist = None
        self.nu_J_dist = None                                  
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.ndot_sources = None                                 # Neutrino number density source terms for integration [cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None                                  # Dirac effective proton mass [MeV]
        self.mn_eff = None                                  # Dirac effective neutron mass [MeV]
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = dU
        self.time = time                                    # Time at which the state occurs during the simulation [s]


class Solver:
    c = 29979245800.0
    HBARC = 1.97327e-11 #MeV * cm

    def __init__(self, table : Table, state : State, opacity_flags : dict, opacity_pars : dict, num_bins : float, nu_external_E : dict):
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        self.num_bins = num_bins
        self.nu_external_E = nu_external_E
        self.E_dist = None


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

    def fermi_dirac_n(self, energy, mu, t):
        return ((1 / (np.exp((energy - mu) / t) + 1))) * ((energy ** 2) / (2 * np.pi ** 2 )) * (1 / (self.HBARC ** 3)) #needs to be divided by HBARC^3 to be in units of cm^-3

    #TODO: NEEDS TESTING
    def calculate_dist(self, state : State):
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, state.mn_eff, state.dm_eff = self.calculate_corrector_quantities(state)

        #Calculate nue and anue chemical potentials
        mu_nue = state.mu_p - state.mu_n + state.mu_e
        mu_anue = - mu_nue

        #construct energy distribution with logarithmic spacing
        E_dist_edges = np.exp(np.linspace(np.log(10 ** -3), np.log(100), self.num_bins + 1))
        E_binwidths = np.diff(E_dist_edges)
        self.E_dist = E_dist_edges[:-1] + (0.5 * E_binwidths)

        #calculate energy and number density distributions using fermi-dirac
        nue_n_dist = self.fermi_dirac_n(self.E_dist, mu_nue, state.t) * E_binwidths
        anue_n_dist = self.fermi_dirac_n(self.E_dist, mu_anue, state.t) * E_binwidths

        nue_J_dist = nue_n_dist * self.E_dist
        anue_J_dist = anue_n_dist * self.E_dist


        return [nue_n_dist.tolist(), anue_n_dist.tolist()], [nue_J_dist.tolist(), anue_J_dist.tolist()]

    #TODO: NEEDS TESTING
    def calculate_spectral_rates(self, state : State):
        #Calculate number and energy density distributions
        state.nu_n_dist, state.nu_J_dist = self.calculate_dist(state)

        #Calculate total number and energy densities by species for bns_nurates
        state.n_m1 = [np.sum(state.nu_n_dist[0]), np.sum(state.nu_n_dist[1]), 0, 0] #nue, anue, nux, anux
        state.J_m1 = [np.sum(state.nu_J_dist[0]), np.sum(state.nu_J_dist[1]), 0, 0] #nue, anue, nux, anux

        #Initialize bns_nurates eos_pars and populate
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = state.nb * 1e-21
        eos_pars.temp = state.t
        eos_pars.ye = state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = state.mu_n
        eos_pars.mu_e = state.mu_e
        eos_pars.mu_p = state.mu_p
        eos_pars.dm_eff = state.dm_eff

        #Set up Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 6
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars

        #Load non-distribution m1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = state.chi_m1

        #Load grey pars
        grey_pars = bns.GreyOpacityParams()
        grey_pars.eos_pars = eos_pars
        grey_pars.opacity_flags = opacity_flags

        rates = []

        #Calculate rates; one set of rates for each energy in E_dist
        for i in range(0, len(self.E_dist)):
            m1_pars.n = [x * 1e-21 for x in state.n_m1]
            m1_pars.J = [x * 1e-21 for x in state.J_m1]

            distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)
            grey_pars.distr_pars = distr_pars
            grey_pars.m1_pars = m1_pars

            spectral_rates = bns.ComputeSpectralOpacitiesStimulatedAbs(self.E_dist[i], quad, grey_pars)
            spectral_rates['kappa'] = [x * 1e7 for x in spectral_rates['kappa']]
            spectral_rates['kappa_s'] = [x * 1e7 for x in spectral_rates['kappa_s']]
            rates.append(spectral_rates)

        return rates

    def calculate_gray_rates(self, state : State):
            """
                Calculates neutrino interaction rates based on the fluid's conditions in the current state.

                Outputs:
                    gray_rates [dict]: A dictionary consisting of energy and number emissivities, energy and number absorptivities, and scattering absorptivities.
            """

            state.t = self.temperature_from_e(state)
            state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
            state.mp_eff, state.mn_eff, state.dm_eff = self.calculate_corrector_quantities(state)

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

            #Populate global structure using grey_pars
            grey_pars = bns.GreyOpacityParams()
            grey_pars.eos_pars = eos_pars
            grey_pars.opacity_flags = opacity_flags
            grey_pars.opacity_pars = opacity_pars
            grey_pars.distr_pars = distr_pars
            grey_pars.m1_pars = m1_pars

            #Calculate rates
            gray_rates = bns.ComputeM1Opacities(quad, quad, grey_pars)
            gray_rates['eta']       = [x * 1e21 for x in gray_rates['eta']]
            gray_rates['eta_0']     = [x * 1e21 for x in gray_rates['eta_0']]
            gray_rates['kappa_a']   = [x * 1e7  for x in gray_rates['kappa_a']]
            gray_rates['kappa_0_a'] = [x * 1e7  for x in gray_rates['kappa_0_a']]
            gray_rates['kappa_s']   = [x * 1e7  for x in gray_rates['kappa_s']]
            return gray_rates

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

nb = (1.01e10 / (eos.unit_dens * eos.mn)) * 1e39 # Baryon number density [cm-3]   
t = 2.17   # Temperature [MeV]
ye   =  0.19

e_nb = np.exp(eos.eval_given_rtx(np.log((eos.thermo["Q7"] + 1) * nb * 1e-39 * eos.mn), np.array([nb]) * 1e-39, np.array([ye]), np.array([t]), method="linear"))[0] * 1e39

n_m1 = [12.56e-3 * nb, 13e-3 * nb, 0, 0]  # Neutrino number densities [cm-3]
J_m1 = [11.89e-2 * nb, 15.56e-2 * nb, 0, 0]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1/3, 1/3, 1/3, 1/3]  # Eddington factor
dU = 3.48e1

init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)
init_state_copy = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': False, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': False, 'use_WM_ab': False, 'use_WM_sc': False, 'use_decay': True, 'brem_implementation': 'GP19', 'neglect_blocking': False, 'use_NN_medium_corr': True}


solver = Solver(eos, init_state, opacity_flags, opacity_pars, 1000, None)
rates = solver.calculate_spectral_rates(init_state)

#calculate the average gray number imfp for each species
nue_kappa = np.array([rate["kappa"][0] for rate in rates])
anue_kappa = np.array([rate["kappa"][1] for rate in rates])

nue_kappa_0 = np.sum(nue_kappa * np.array(init_state.nu_n_dist[0])) / np.sum(np.array(init_state.nu_n_dist[0]))
anue_kappa_0 = np.sum(anue_kappa * np.array(init_state.nu_n_dist[1])) / np.sum(np.array(init_state.nu_n_dist[1]))

gray_rates = solver.calculate_gray_rates(init_state_copy)
print("--------------------------------------")
print(f"nue number imfp ratio [gray calculated from spectral vs / bns_nurates gray]: {nue_kappa_0 / (gray_rates["kappa_0_a"][0])}")
print(f"anue number imfp ratio [gray calculated from spectral / bns_nurates gray]: {anue_kappa_0 / (gray_rates["kappa_0_a"][1])}")
print("--------------------------------------")




