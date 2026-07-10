import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from os import system, name
from scipy.optimize import bisect
import pandas as pd


class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1, time = 0.0):
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
        self.time = time                                    # Time at which the state occurs during the simulation [s]

class Solver:
    c = 29979245800.0

    def __init__(self, table : Table, state : State, integrate_tolerance : float, opacity_flags : dict, opacity_pars : dict):
        self.timestep = None
        self.integrate_tolerance = integrate_tolerance
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars


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
            return np.log(state.fluid_e * 1e-39) - e_interp

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

    def calculate_gray_rates(self, state : State):
        """
            Calculates neutrino interaction rates based on the fluid's conditions in the current state.

            Outputs:
                gray_rates [dict]: A dictionary consisting of energy and number emissivities, energy and number absorptivities, and scattering absorptivities.
        """

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

    def calculate_e_source_terms(self, state : State):
        """
            Calculates the neutrino energy density source terms needed for numerical integration based on the conditions in the current simulation state.

            Inputs:
                J_m1 [list]: list of neutrino energy densities sorted by species in the format [nue, anue, nux, anux]
            
            Outputs:
                [numpy.ndarray]: A numpy array containing the energy density source terms in the format [nue, anue, nux, anux]
        """

        rates = state.rates
        e_terms = [None, None, None, None]

        for i in range(0, 4):
            e_terms[i] = rates["eta"][i] - (self.c * rates["kappa_a"][i] * state.J_m1[i])

        return np.array(e_terms)
    
    def calculate_n_source_terms(self, state : State):
        """
            Calculates the neutrino number density source terms needed for numerical integration based on the conditions in the current simulation state.

            Inputs:
                n_m1 [list]: list of neutrino number densities sorted by species in the format [nue, anue, nux, anux]
            
            Outputs:
                [numpy.ndarray]: A numpy array containing the energy density source terms in the format [nue, anue, nux, anux]
        """
        rates = state.rates
        n_terms = [None, None, None, None]
        
        for i in range(0, 4):
            n_terms[i] = rates["eta_0"][i] - (self.c * rates["kappa_0_a"][i] * state.n_m1[i])

        return np.array(n_terms)
    
    def calculate_state(self, state : State):
        """
            Calculates state variables left blank at the time of state creation. Updates the inputted state in-place.
        """
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, state.mn_eff, state.dm_eff = self.calculate_corrector_quantities(state)
        state.rates = self.calculate_gray_rates(state)
        state.edot_sources = self.calculate_e_source_terms(state)
        state.ndot_sources = self.calculate_n_source_terms(state)
        return

    def generate_new_state(self, nb : float, ye : float, fluid_e : float, n_m1 : list, J_m1 : list, time : float):
        """
            Given a number density, electron fraction, and energy density, creates a new State object.

            Inputs:
                nb [float]: The fluid's baryon number density [cm-3].
                ye [float]: The fluid's electron fraction.
                fluid_e [float]: The fluid's energy density [MeV cm-3]
                time [float]: The current time in the simulation for the new state.
            
            Outputs:
                [State]: A new state object containing the inputted values. n_m1, J_m1, and chi_m1 from the previous state persist into the new State object.
        """
        return State(nb, ye, fluid_e, n_m1, J_m1, [1/3, 1/3, 1/3, 1/3], time)

    def integrate_step(self, state : State):
        """
            Integrates the simulation forward in time by one timestep. Finds the next fluid energy density and electron fraction using a variable step size RK2 integrator, where the timestep corresponds
            to a small fraction of the absorptivity corresponding to the shortest timescale.

            Outputs:
                next_e [float]: The fluid's energy density at the next timestep of the simulation.
                next_ye [float]: The fluid's electron fraction at the next timestep of the simulation.
                next_time [float]: The simulation time at the next timestep.
        """

        # Calculate timestep, largest absorptivity corresponds to shortest timescale
        self.calculate_state(self.states[-1])

        timestep = 0.01 * ((1 / max(state.rates["kappa_a"] + state.rates["kappa_0_a"])) / self.c)
        
        # Find k1 for neutrino fields and matter fields
        k1_J = (timestep * np.array(state.edot_sources))
        k1_n = (timestep * np.array(state.ndot_sources))

        #Calculate k1 for the fluid energy density and electron fraction
        k1_e = -(np.sum(state.edot_sources)) * timestep
        k1_ye = ((state.ndot_sources[1] - state.ndot_sources[0]) / state.nb) * timestep

        #Calculate state at midpoint of timestep
        e_fluid_mid = state.fluid_e + (k1_e / 2)
        ye_mid = state.ye +  (k1_ye / 2)

        J_m1_mid = np.array(state.J_m1) + (k1_J / 2)
        n_m1_mid = np.array(state.n_m1) + (k1_n / 2)

        midpoint = self.generate_new_state(state.nb, ye_mid, e_fluid_mid, n_m1_mid, J_m1_mid, state.time + (timestep * 0.5))

        self.calculate_state(midpoint)

        #Find k2 for neutrino fields and matter fields
        k2_J = (timestep * np.array(midpoint.edot_sources))
        k2_n = (timestep * np.array(midpoint.ndot_sources))

        #Calculate k2 for the fluid energy density and electron fraction
        k2_e = -(np.sum(midpoint.edot_sources)) * timestep
        k2_ye = ((midpoint.ndot_sources[1] - midpoint.ndot_sources[0]) / midpoint.nb) * timestep

        #Update neutrino fields
        new_J_m1 = np.array(state.J_m1) + k2_J
        new_n_m1 = np.array(state.n_m1) + k2_n

        #Update fluid energy density and electron fraction
        new_fluid_e = state.fluid_e + k2_e
        new_ye = state.ye + k2_ye

        #Generate next state
        next_state = self.generate_new_state(state.nb, new_ye, new_fluid_e, new_n_m1, new_J_m1, state.time + timestep)

        return next_state
    
    def integrate_to_eq(self, verbose=False):
        eq_detected = False
        num_states = 1
        while not eq_detected:
            next_state = self.integrate_step(self.states[-1])
            
            if verbose:
                print(f"\nTime: {self.states[-1].time}"
                    f"\ne: {self.states[-1].fluid_e}"
                    f"\nYe: {self.states[-1].ye}"
                    f"\nT: {self.states[-1].t}"
                    f"\nnb: {self.states[-1].nb}"
                    f"\nmu_p: {self.states[-1].mu_p}"
                    f"\nmu_n: {self.states[-1].mu_n}"
                    f"\nmu_e: {self.states[-1].mu_e}"
                    f"\nmn_eff: {self.states[-1].mn_eff}"
                    f"\nmp_eff: {self.states[-1].mp_eff}"
                    f"\ndm_eff: {self.states[-1].dm_eff}"
                    f"\nedot_sources: {self.states[-1].edot_sources.tolist()}"
                    f"\nndot_sources: {self.states[-1].ndot_sources.tolist()}",
                    f"\nn_m1: {self.states[-1].n_m1}",
                    f"\nJ_m1: {self.states[-1].J_m1}",
                    f"\nchi_m1: {self.states[-1].chi_m1}",
                    f"\nNumber of saved states: {num_states}")

            self.states.append(next_state)
            num_states += 1

            # Check to see if numerical integration has converged for both e_fluid and ye. Integration is considered to converge if the difference between the last state and the new state is less than the solver's integration tolerance.
            if (np.abs(self.states[-1].ye - self.states[-2].ye) < self.integrate_tolerance):
                eq_detected = True
        self.calculate_state(self.states[-1])
        self.save_to_csv()
        return

    def save_to_csv(self):
        nb_list = []
        e_list = []
        ye_list = []
        t_list = []
        edot_sources_list = []
        ndot_sources_list = []
        n_m1_list = []
        J_m1_list = []
        time_list = []
        
        for state in self.states:
            nb_list.append(state.nb)
            e_list.append(state.fluid_e)
            ye_list.append(state.ye)
            t_list.append(state.t)
            edot_sources_list.append(state.edot_sources)
            ndot_sources_list.append(state.ndot_sources)
            n_m1_list.append(state.n_m1)
            J_m1_list.append(state.J_m1)
            time_list.append(state.time)

        data = pd.DataFrame({"Time" : time_list, "nb" : nb_list, "e_fluid" : e_list, "ye" : ye_list, "t" : t_list, "edot_sources" : edot_sources_list, "ndot_sources" : ndot_sources_list, "n_m1" : n_m1_list, "J_m1" : J_m1_list})
        data.to_csv("single_zone.csv")
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

## Input thermodynamic quantities (corresponding to point F in Chiesa+25 PRD)
##  N.B.: chemical potentials include the rest mass contribution
nb   = (1.0182011974e+10 / eos.unit_dens) * 1e39 # Baryon number density [cm-3]      
t = 2.1441581249e+00    # Temperature [MeV]
ye   = 1.8740089238e-01    # Electron fraction
xn   = 7.9639333255e-01                # Neutron fraction
xp   = 1.7166997157e-01                     # Proton fraction
e_nb = np.exp(eos.eval_given_rtx(np.log((eos.thermo["Q7"] + 1) * nb * 1e-39 * eos.mn), np.array([nb]) * 1e-39, np.array([ye]), np.array([t])))[0] * 1e39

n_m1 = [1.2163992294e-02 * nb, 1.1021108202e-02 * nb, 4.6724985033e-04 * nb, 4.6724985033e-04 * nb]  # Neutrino number densities [cm-3]
J_m1 = [1.0770866527e-01 * nb,  1.3601605129e-01 * nb,  9.7433536159e-03 * nb,  9.7433536159e-03 * nb]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1/3, 1/3, 1/3, 1/3]  # Eddington factor

print(f"nb: {nb} cm-3",
      f"\nt: {t} MeV",
      f"\nye: {ye}",
      f"\nxn: {xn}",
      f"\nxp: {xp}",
      f"\ne_nb: {e_nb} MeV cm-3",
      f"\nn_m1: {n_m1} cm-3",
      f"\nJ_m1: {J_m1} MeV cm-3",
      f"\nchi_m1: {chi_m1}")

init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': True, 'use_WM_ab': True, 'use_WM_sc': True, 'use_decay': True, 'brem_implementation': 'HR98', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state, 1e-8, opacity_flags, opacity_pars)
solver.calculate_state(init_state)
"""print(f"\nTime: {init_state.time}"
                    f"\ne: {init_state.fluid_e}"
                    f"\nYe: {init_state.ye}"
                    f"\nT: {init_state.t}"
                    f"\nnb: {init_state.nb}"
                    f"\nmu_p: {init_state.mu_p}"
                    f"\nmu_n: {init_state.mu_n}"
                    f"\nmu_e: {init_state.mu_e}"
                    f"\nmn_eff: {init_state.mn_eff}"
                    f"\nmp_eff: {init_state.mp_eff}"
                    f"\ndm_eff: {init_state.dm_eff}"
                    f"\nedot_sources: {init_state.edot_sources.tolist()}"
                    f"\nndot_sources: {init_state.ndot_sources.tolist()}",
                    f"\nn_m1: {init_state.n_m1}",
                    f"\nJ_m1: {init_state.J_m1}",
                    f"\nchi_m1: {init_state.chi_m1}")"""


#solver.integrate_to_eq(verbose=True)




