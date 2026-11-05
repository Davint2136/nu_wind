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

    def __init__(self, table : Table, state : State, opacity_flags : dict, opacity_pars : dict, binwidth : float, nu_external_E : dict):
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        self.E_binwidth = binwidth
        #This is a linear energy range, and will result in linear distributions. I need to construct them logarithmically. The distribution yields higher occupation numbers for lower energies, so linear isn't a good fit here.
        #I'll leave this for now.
        self.E_range = np.arange(0.1, 100, self.E_binwidth)
        self.nu_external_E = nu_external_E
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

    #TODO: NEEDS TESTING
    def calculate_dist(self, state : State):
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, mn_eff, dm_eff = self.calculate_corrector_quantities(state)

        #Calculate nue and anue chemical potentials
        mu_nue = state.mu_p - state.mu_n + state.mu_e
        mu_anue = - mu_nue

        #calculate energy and number density distributions using fermi-dirac
        nue_n_dist = ((1 / (2 * np.pi ** 2)) * ((self.E_range ** 2) / (np.exp((self.E_range - mu_nue) / state.t) + 1))) * self.E_binwidth
        anue_n_dist = ((1 / (2 * np.pi ** 2)) * ((self.E_range ** 2) / (np.exp((self.E_range - mu_anue) / state.t) + 1))) * self.E_binwidth

        nue_J_dist = ((1 / (2 * np.pi ** 2)) * ((self.E_range ** 3) / (np.exp((self.E_range - mu_nue) / state.t) + 1))) * self.E_binwidth
        anue_J_dist = ((1 / (2 * np.pi ** 2)) * ((self.E_range ** 3) / (np.exp((self.E_range - mu_anue) / state.t) + 1))) * self.E_binwidth

        nu_n_dist = [nue_n_dist.tolist(), anue_n_dist.tolist()]
        nu_J_dist = [nue_J_dist.tolist(), anue_J_dist.tolist()]

        return nu_n_dist, nu_J_dist

    #TODO: NEEDS TESTING
    def calculate_spectral_rates(self, state : State):
        #Calculate number and energy density distributions
        state.nu_n_dist, state.nu_J_dist = self.calculate_dist(state)

        #Initialize bns_nurates eos_pars and populate
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = state.nb
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

        for i in range(0, len(self.E_range)):
            m1_pars.n = [x * 1e-21 for x in state.nu_n_dist[i]]
            m1_pars.J = [x * 1e-21 for x in state.nu_J_dist[i]]

            distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)
            grey_pars.distr_pars = distr_pars
            grey_pars.m1_pars = m1_pars

            spectral_rates = bns.ComputeSpectralOpacitiesStimulatedAbs(self.E_range[i], quad, grey_pars)
            spectral_rates['kappa'] = [x * 1e7 for x in spectral_rates['kappa']]
            spectral_rates['kappa_s'] = [x * 1e7 for x in spectral_rates['kappa_s']]
            rates.append(spectral_rates)

        return rates


#TESTING-------------------------------------------------------------
    def bnsnurates_dist(self, state : State):
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, mn_eff, dm_eff = self.calculate_corrector_quantities(state)
    
        #Calculate nue and anue chemical potentials
        mu_nue = state.mu_p - state.mu_n + state.mu_e
        mu_anue = - mu_nue

        #Calculate distributions: bns.FermiDistr() only gives the fermi distribution, so appropriate scaling terms must be included to calculate the desired distributions.
        nu_n_dist = [[(1 / (2 * np.pi ** 2)) * bns.FermiDistr(energy, state.t, mu_nue) * (energy ** 2) * self.E_binwidth for energy in self.E_range], 
                     [(1 / (2 * np.pi ** 2)) * bns.FermiDistr(energy, state.t, mu_anue) * (energy ** 2) * self.E_binwidth for energy in self.E_range]]
        
        nu_J_dist = [[(1 / (2 * np.pi ** 2)) * bns.FermiDistr(energy, state.t, mu_nue) * (energy ** 3) * self.E_binwidth for energy in self.E_range],
                     [(1 / (2 * np.pi ** 2)) * bns.FermiDistr(energy, state.t, mu_anue) * (energy ** 3) * self.E_binwidth for energy in self.E_range]]

        return nu_n_dist, nu_J_dist
        



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

n_m1 = [0.36e-3 * nb, 2.21e-3 * nb, 0, 0]  # Neutrino number densities [cm-3]
J_m1 = [0.18e-1 * nb, 1.23e-1 * nb, 0, 0]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1/3, 1/3, 1/3, 1/3]  # Eddington factor
dU = 3.48e1

init_state1 = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)
init_state2 = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': False, 'use_WM_ab': False, 'use_WM_sc': False, 'use_decay': True, 'brem_implementation': 'GP19', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state1, opacity_flags, opacity_pars, 1, None)

#Calculate distributions using my function and bns_nurates' FermiDistr() function
nu_n_dist, nu_e_dist = solver.calculate_dist(init_state1)
bns_nu_n_dist, bns_nu_e_dist = solver.bnsnurates_dist(init_state2)

#Calculate errors
nu_n_error_dist = [np.abs(np.array(nu_n_dist[0]) - np.array(bns_nu_n_dist[0])),
                   np.abs(np.array(nu_n_dist[1]) - np.array(bns_nu_n_dist[1]))]

nu_e_error_dist = [np.abs(np.array(nu_e_dist[0]) - np.array(bns_nu_e_dist[0])),
                   np.abs(np.array(nu_e_dist[1]) - np.array(bns_nu_e_dist[1]))]

print(f"NUE_N_ERROR:\n{nu_n_error_dist[0]}\n")
print(f"ANUE_N_ERROR:\n{nu_n_error_dist[1]}\n")

print(f"NUE_E_ERROR:\n{nu_e_error_dist[0]}\n")
print(f"ANUE_E_ERROR:\n{nu_e_error_dist[1]}\n")

plt.figure()
plt.plot(solver.E_range, nu_n_error_dist[0], label=r'$\nu_e$')
plt.plot(solver.E_range, nu_n_error_dist[1], label=r'$\bar{\nu}_e$')
plt.xlabel("Energy [MeV]")
plt.ylabel(r"$n_{\nu_\times}$ Error")
plt.title("Neutrino Number Density Error vs. bns_nurates Implementation")
plt.legend()
plt.savefig("nu_n_error.png")
plt.show()

plt.figure()
plt.plot(solver.E_range, nu_e_error_dist[0], label=r"$\nu_e$")
plt.plot(solver.E_range, nu_e_error_dist[1], label=r"$\bar{\nu}_e$")
plt.xlabel("Energy [MeV]")
plt.ylabel(r"$e_{\nu_\times}$ Error")
plt.title("Neutrino Energy Density Error vs. bns_nurates Implementation")
plt.legend()
plt.savefig("nu_e_error.png")
plt.show()